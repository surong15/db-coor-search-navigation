#!/usr/bin/env python3
# 拿掉clip，用llava對資料庫的ai answer做篩選挑出候選人，再用llava對圖片及敘述做評估
# 改用ROS Bridge WebSocket與ros2_navigation extension通訊

"""
機器人視覺記錄搜尋系統 - LLaVA語義搜尋版本 + ROS Bridge整合
根據自然語言指令從Milvus資料庫中搜尋最符合的目標座標，並使用LLaVA進行精確評分
使用關鍵字篩選 + LLaVA視覺評估，不依賴CLIP模型
透過ROS Bridge WebSocket與ros2_navigation extension進行機器人導航控制
"""

import numpy as np
import base64
from io import BytesIO
from PIL import Image, ImageTk
from pymilvus import Collection, connections
from typing import List, Dict, Any, Optional
import json
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
from datetime import datetime
import requests
import tempfile
import os
import traceback
import time
from enum import Enum
from abc import ABC, abstractmethod

# 檢查 Qdrant 是否可用
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
    QDRANT_AVAILABLE = True
    logger_qdrant = True
except ImportError:
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "qdrant-client"])
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
        QDRANT_AVAILABLE = True
        logger_qdrant = True
    except:
        QDRANT_AVAILABLE = False
        logger_qdrant = False

# 檢查 WebSocket 是否可用 (用於 ROS Bridge)
try:
    import websocket
    import ssl
    WEBSOCKET_AVAILABLE = True
    logger_websocket = True
except ImportError:
    try:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websocket-client"])
        import websocket
        import ssl
        WEBSOCKET_AVAILABLE = True
        logger_websocket = True
    except:
        WEBSOCKET_AVAILABLE = False
        logger_websocket = False

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if logger_websocket and WEBSOCKET_AVAILABLE:
    logger.info("WebSocket 客戶端可用，支援 ROS Bridge 通訊")
else:
    logger.warning("WebSocket 客戶端不可用，僅支援 HTTP 通訊")

if logger_qdrant and QDRANT_AVAILABLE:
    logger.info("Qdrant 客戶端可用，支援 Qdrant 向量資料庫")
else:
    logger.warning("Qdrant 客戶端不可用，僅支援 Milvus 資料庫")

# 資料庫類型枚舉
class DatabaseType(Enum):
    MILVUS = "milvus"
    QDRANT = "qdrant"

# 抽象搜尋器基類
class BaseVisionSearcher(ABC):
    """視覺搜尋器抽象基類"""
    
    @abstractmethod
    def search_candidates(self, instruction: str, top_k: int = 50, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """搜尋候選位置"""
        pass
    
    @abstractmethod
    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """解碼base64影像"""
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """獲取資料庫集合資訊"""
        pass

class IsaacSimClient:
    def __init__(self, base_url: str = "ws://localhost:9090"):
        """
        初始化Isaac Sim ROS Bridge客戶端
        
        Args:
            base_url: ROS Bridge WebSocket 服務器地址
        """
        self.rosbridge_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
        if not self.rosbridge_url.startswith(("ws://", "wss://")):
            self.rosbridge_url = f"ws://{base_url}" if "://" not in base_url else base_url
        
        self.ws = None
        self.connected = False
        self.navigation_topic = "/baymax/navigation_goal"
        self.status_topic = "/baymax/navigation_status"
        self.last_navigation_time = 0
        self.min_interval = 0  # 移除導航間隔限制
        self.navigation_status = "Idle"
        self.connection_thread = None
        
        logger.info(f"初始化 ROS Bridge 客戶端: {self.rosbridge_url}")
    
    def test_connection(self) -> bool:
        """測試與ROS Bridge的連接"""
        if not WEBSOCKET_AVAILABLE:
            logger.error("WebSocket 不可用，無法連接 ROS Bridge")
            return False
            
        try:
            # 嘗試建立短暫連接測試
            test_ws = websocket.create_connection(self.rosbridge_url, timeout=5)
            test_ws.close()
            return True
        except Exception as e:
            logger.error(f"ROS Bridge連接測試失敗: {e}")
            return False
    
    def connect(self):
        """連接到 ROS Bridge"""
        if not WEBSOCKET_AVAILABLE:
            logger.error("WebSocket 不可用，無法連接")
            return False
            
        try:
            logger.info(f"正在連接到 ROS Bridge: {self.rosbridge_url}")
            
            self.ws = websocket.WebSocketApp(
                self.rosbridge_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # 在背景執行緒中運行
            self.connection_thread = threading.Thread(target=self._run_websocket, daemon=True)
            self.connection_thread.start()
            
            # 等待連接建立
            for _ in range(50):  # 最多等5秒
                if self.connected:
                    break
                time.sleep(0.1)
            
            return self.connected
            
        except Exception as e:
            logger.error(f"連接 ROS Bridge 失敗: {e}")
            return False
    
    def _run_websocket(self):
        """在背景執行緒中運行 WebSocket"""
        try:
            self.ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            logger.error(f"WebSocket 執行錯誤: {e}")
            self.connected = False
    
    def _on_open(self, ws):
        """WebSocket 連接建立"""
        self.connected = True
        logger.info("✅ ROS Bridge 已連接")
        
        # 廣告導航 topic
        self._advertise_topics()
        # 訂閱狀態 topic
        self._subscribe_status()
    
    def _on_close(self, ws, close_status_code, close_msg):
        """WebSocket 連接關閉"""
        self.connected = False
        logger.info(f"❌ ROS Bridge 連接關閉: {close_status_code}, {close_msg}")
    
    def _on_error(self, ws, error):
        """WebSocket 錯誤"""
        self.connected = False
        logger.error(f"❌ ROS Bridge 錯誤: {error}")
    
    def _on_message(self, ws, message):
        """接收 WebSocket 訊息"""
        try:
            data = json.loads(message)
            
            if data.get("op") == "publish" and data.get("topic") == self.status_topic:
                # 接收導航狀態更新
                msg = data.get("msg", {})
                self.navigation_status = msg.get("data", "Unknown")
                logger.debug(f"導航狀態更新: {self.navigation_status}")
                
        except Exception as e:
            logger.error(f"處理 ROS 訊息失敗: {e}")
    
    def _advertise_topics(self):
        """廣告 ROS topics"""
        topics = [
            {
                "op": "advertise",
                "topic": self.navigation_topic,
                "type": "geometry_msgs/PoseStamped"
            }
        ]
        
        for topic_config in topics:
            if self.connected and self.ws:
                self.ws.send(json.dumps(topic_config))
                logger.info(f"📢 廣告 topic: {topic_config['topic']}")
                time.sleep(0.1)
    
    def _subscribe_status(self):
        """訂閱狀態 topic"""
        status_config = {
            "op": "subscribe",
            "topic": self.status_topic,
            "type": "std_msgs/String"
        }
        
        if self.connected and self.ws:
            self.ws.send(json.dumps(status_config))
            logger.info(f"訂閱狀態 topic: {self.status_topic}")
    
    def get_status(self) -> Dict[str, Any]:
        """獲取導航狀態"""
        if not self.connected:
            return {'success': False, 'error': 'ROS Bridge 未連接'}
        
        return {
            'success': True,
            'robot_name': 'Baymax',
            'navigation_status': self.navigation_status,
            'connected': self.connected,
            'timestamp': time.time()
        }
    
    def navigate_to_position(self, position: Dict[str, float], force: bool = False) -> Dict[str, Any]:
        """
        透過 ROS Bridge 發送導航指令
        
        Args:
            position: 目標位置座標 {'x': float, 'y': float, 'z': float}
            force: 是否強制發送（忽略時間間隔限制）
            
        Returns:
            導航結果字典
        """
        if not WEBSOCKET_AVAILABLE:
            return {
                'success': False,
                'error': 'WebSocket 不可用，請安裝 websocket-client'
            }
        
        if not self.connected:
            # 嘗試重新連接
            if not self.connect():
                return {
                    'success': False,
                    'error': 'ROS Bridge 未連接且無法建立連接'
                }
        
        try:
            current_time = time.time()
            
            # 構建 PoseStamped 訊息
            pose_msg = {
                "op": "publish",
                "topic": self.navigation_topic,
                "msg": {
                    "header": {
                        "stamp": {
                            "sec": int(current_time),
                            "nanosec": int((current_time % 1) * 1e9)
                        },
                        "frame_id": "world"
                    },
                    "pose": {
                        "position": {
                            "x": float(position['x']),
                            "y": float(position['y']),
                            "z": float(position['z'])
                        },
                        "orientation": {
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "w": 1.0
                        }
                    }
                }
            }
            
            # 發送 ROS 訊息
            if self.connected and self.ws:
                self.ws.send(json.dumps(pose_msg))
                self.last_navigation_time = current_time
                
                logger.info(f"✅ ROS 導航指令已發送: ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})")
                
                return {
                    'success': True,
                    'message': f'ROS 導航指令已發送到位置: {position}',
                    'method': 'ROS Bridge WebSocket',
                    'topic': self.navigation_topic
                }
            else:
                return {
                    'success': False,
                    'error': 'WebSocket 連接中斷'
                }
                
        except Exception as e:
            logger.error(f"ROS 導航指令發送失敗: {e}")
            return {
                'success': False,
                'error': f'ROS 發送失敗: {str(e)}'
            }
    
    def disconnect(self):
        """斷開連接"""
        self.connected = False
        if self.ws:
            self.ws.close()
        logger.info("ROS Bridge 已斷開連接")
    
    def get_time_until_next_navigation(self) -> float:
        """獲取距離下次可導航的剩餘時間"""
        current_time = time.time()
        elapsed = current_time - self.last_navigation_time
        remaining = max(0, self.min_interval - elapsed)
        return remaining

class OllamaLLaVAClient:
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llava"):
        """
        初始化Ollama LLaVA客戶端
        
        Args:
            base_url: Ollama服務器地址
            model_name: LLaVA模型名稱
        """
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.generate_url = f"{self.base_url}/api/generate"
    
    def test_connection(self) -> bool:
        """測試與Ollama的連接"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def evaluate_candidate(self, instruction: str, image_base64: str, ai_description: str, 
                          position: Dict[str, float]) -> Dict[str, Any]:
        """
        使用LLaVA評估候選位置
        
        Args:
            instruction: 用戶指令
            image_base64: 候選位置的影像(base64)
            ai_description: AI對該位置的描述
            position: 位置座標
            
        Returns:
            評分結果字典
        """
        try:
            # 構建評分prompt
            prompt = self._build_evaluation_prompt(instruction, ai_description, position)
            
            # 準備請求數據
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False,
                "options": {
                    "temperature": 0.2,  # 降低隨機性以獲得一致的評分
                    "top_p": 0.9
                }
            }
            
            # 發送請求
            response = requests.post(self.generate_url, json=request_data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            llava_response = result.get('response', '')
            
            # 解析LLaVA的回應
            parsed_result = self._parse_llava_response(llava_response)
            
            return {
                'success': True,
                'score': parsed_result.get('score', 0),
                'reasoning': parsed_result.get('reasoning', ''),
                'raw_response': llava_response,
                'confidence': parsed_result.get('confidence', 'medium')
            }
            
        except Exception as e:
            logger.error(f"LLaVA評估失敗: {e}")
            return {
                'success': False,
                'score': 0,
                'reasoning': f'評估失敗: {str(e)}',
                'raw_response': '',
                'confidence': 'low'
            }
    
    def _build_evaluation_prompt(self, instruction: str, ai_description: str, position: Dict[str, float]) -> str:
        """構建評分prompt - 平衡結合影像觀察與AI描述"""
        prompt = f"""你是一個專業的機器人路徑規劃評估專家。請根據以下資訊評估這個位置與用戶指令的「符合程度」，並給出0~100的分數。

**用戶指令:** {instruction}

**位置座標:** ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})

**AI過去對此位置的描述:** {ai_description}

**評估重點說明:**
- **主要任務**: 仔細觀察提供的影像，判斷畫面內容是否符合用戶指令
- **輔助參考**: 結合AI過去的描述進行交叉驗證
- **評估原則**: 影像觀察(70%) + AI描述驗證(30%)

**詳細評估步驟:**
1. **影像分析 (主要依據)**:
   - 仔細觀察影像中的所有物體、顏色、形狀、環境
   - 識別與用戶指令相關的視覺元素
   - 評估影像內容與指令的直接匹配度

2. **描述驗證 (輔助參考)**:
   - 檢查AI描述是否與你觀察到的影像內容一致
   - 如果描述與影像不符，以影像觀察為準
   - 如果描述與影像相符，可作為補充判斷依據

3. **綜合評分標準**:
   - **90-100分**: 影像中清楚顯示指令要求的物體/環境，且AI描述相符
   - **70-89分**: 影像中能看到相關物體/環境，但可能不完全符合或不夠清晰
   - **50-69分**: 影像中有部分相關元素，但不是主要特徵
   - **30-49分**: 影像中只有微弱相關性，或描述提到但影像不明顯
   - **0-29分**: 影像與指令基本無關，無論描述如何

**評估範例:**
- 用戶指令"找到紅色物體"：
  * 影像清楚顯示紅色椅子 + 描述提到"紅色椅子" → 95分
  * 影像顯示紅色物體但描述沒提到 → 85分
  * 描述提到紅色但影像看不清楚 → 45分

- 用戶指令"走到桌子那"：
  * 影像顯示清晰的桌子 + 描述提到"辦公桌" → 90分
  * 影像顯示桌子但描述說是"工作區域" → 80分
  * 描述提到桌子但影像看不到 → 25分

**重要提醒:**
- 請務必仔細觀察影像，不要只依賴AI描述
- 如果影像與描述有衝突，以你的影像觀察為準
- 評分要反映影像內容與指令的真實匹配程度

**請用以下格式回答:**
評分: [0~100的整數]
置信度: [high/medium/low]
理由: [先描述你在影像中看到的內容，再說明與AI描述的對比，最後解釋評分依據]

請開始評估:"""
        return prompt
    
    def _parse_llava_response(self, response: str) -> Dict[str, Any]:
        """解析LLaVA的回應"""
        try:
            lines = response.strip().split('\n')
            result = {
                'score': 0,
                'confidence': 'medium',
                'reasoning': response  # 預設使用完整回應作為理由
            }
            
            for line in lines:
                line = line.strip()
                if line.startswith('評分:') or line.startswith('分數:') or line.startswith('Score:'):
                    # 提取分數
                    import re
                    numbers = re.findall(r'\d+', line)
                    if numbers:
                        score = int(numbers[0])
                        result['score'] = max(0, min(100, score))  # 確保分數在0-100範圍內
                
                elif line.startswith('置信度:') or line.startswith('Confidence:'):
                    # 提取置信度
                    if 'high' in line.lower():
                        result['confidence'] = 'high'
                    elif 'low' in line.lower():
                        result['confidence'] = 'low'
                    else:
                        result['confidence'] = 'medium'
                
                elif line.startswith('理由:') or line.startswith('Reason:'):
                    # 提取理由
                    reasoning = line.split(':', 1)[1].strip()
                    if reasoning:
                        result['reasoning'] = reasoning
            
            return result
            
        except Exception as e:
            logger.error(f"解析LLaVA回應失敗: {e}")
            return {
                'score': 0,
                'confidence': 'low',
                'reasoning': f'解析失敗: {response}'
            }

class RobotVisionSearcher(BaseVisionSearcher):
    def __init__(self, 
                 milvus_host: str = "localhost", 
                 milvus_port: str = "19530",
                 collection_name: str = "ros2_camera_images"):
        """初始化Milvus搜尋系統"""
        try:
            self.database_type = DatabaseType.MILVUS
            self.collection_name = collection_name
            self.collection = None
            
            logger.info(f"初始化 Milvus RobotVisionSearcher")
            
            # 連接Milvus
            self._connect_milvus(milvus_host, milvus_port)
            
            # 載入Collection
            self._load_collection()
            
            logger.info("Milvus RobotVisionSearcher 初始化完成")
            
        except Exception as e:
            logger.error(f"Milvus RobotVisionSearcher 初始化失敗: {e}")
            raise
    
    def _connect_milvus(self, host: str, port: str):
        """連接到Milvus"""
        try:
            connections.connect("default", host=host, port=port)
            logger.info(f"成功連接到Milvus: {host}:{port}")
        except Exception as e:
            logger.error(f"連接Milvus失敗: {e}")
            raise
    
    def _load_collection(self):
        """載入Collection"""
        try:
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"成功載入Collection: {self.collection_name}")
            
            stats = self.collection.num_entities
            logger.info(f"Collection包含 {stats} 筆記錄")
            
        except Exception as e:
            logger.error(f"載入Collection失敗: {e}")
            raise
    
    def search_candidates(self, instruction: str, top_k: int = 50, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """根據指令搜尋候選位置 - 使用關鍵字篩選ai_answer欄位"""
        try:
            logger.info(f"正在搜尋指令: '{instruction}'，使用關鍵字篩選ai_answer欄位")
            
            # 提取指令中的關鍵字
            keywords = self._extract_keywords(instruction)
            logger.info(f"提取的關鍵字: {keywords}")
            
            # 設定分頁參數，先獲取更多記錄用於篩選
            initial_limit = min(500, top_k * 10)  # 獲取更多記錄用於篩選
            
            results = self.collection.query(
                expr="",  # 空字符串表示獲取所有記錄
                output_fields=[
                    "camera_path", 
                    "image_base64", 
                    "ai_question", 
                    "ai_answer",
                    "position_x", 
                    "position_y", 
                    "position_z",
                    "rotation_x",
                    "rotation_y", 
                    "rotation_z",
                    "rotation_w",
                    "metadata",
                    "timestamp"
                ],
                limit=initial_limit
            )
            
            logger.info(f"從資料庫獲取了 {len(results)} 筆記錄，目標需要 {top_k} 筆候選")
            
            # 根據關鍵字篩選候選，分為兩類
            matched_candidates = []  # 有關鍵字匹配的候選
            all_candidates = []      # 所有候選（包括無匹配的）
            
            for entity in results:
                ai_answer = entity.get('ai_answer', '').lower()
                
                # 計算關鍵字匹配度
                match_score = self._calculate_keyword_match(ai_answer, keywords, instruction.lower())
                
                candidate = {
                    'id': entity.get('id'),
                    'keyword_match_score': match_score,  # 關鍵字匹配分數
                    'llava_score': None,  # 添加LLaVA分數欄位
                    'llava_reasoning': None,  # 添加LLaVA理由欄位
                    'llava_confidence': None,  # 添加LLaVA置信度欄位
                    'image_base64': entity.get('image_base64'),
                    'camera_path': entity.get('camera_path'),
                    'ai_answer': entity.get('ai_answer'),
                    'ai_question': entity.get('ai_question'),
                    'position': {
                        'x': float(entity.get('position_x', 0)),
                        'y': float(entity.get('position_y', 0)),
                        'z': float(entity.get('position_z', 0))
                    },
                    'rotation': {
                        'x': float(entity.get('rotation_x', 0)),
                        'y': float(entity.get('rotation_y', 0)),
                        'z': float(entity.get('rotation_z', 0)),
                        'w': float(entity.get('rotation_w', 1))
                    },
                    'metadata': entity.get('metadata'),
                    'timestamp': entity.get('timestamp')
                }
                
                if match_score > 0:
                    matched_candidates.append(candidate)
                else:
                    all_candidates.append(candidate)
            
            # 先排序匹配的候選
            matched_candidates.sort(key=lambda x: x['keyword_match_score'], reverse=True)
            
            # 組合最終結果：優先使用匹配的候選，不足時補充其他候選
            final_candidates = []
            
            # 添加匹配的候選
            final_candidates.extend(matched_candidates[:top_k])
            
            # 如果匹配的候選不夠，補充其他候選
            if len(final_candidates) < top_k:
                additional_needed = top_k - len(final_candidates)
                
                # 從未匹配的候選中補充（隨機打亂以增加多樣性）
                import random
                random.shuffle(all_candidates)
                
                # 確保不重複添加
                existing_ids = {c['id'] for c in final_candidates}
                for candidate in all_candidates:
                    if candidate['id'] not in existing_ids and len(final_candidates) < top_k:
                        final_candidates.append(candidate)
                
                logger.info(f"關鍵字匹配 {len(matched_candidates)} 筆，補充 {len(final_candidates) - len(matched_candidates)} 筆，共 {len(final_candidates)} 筆")
            else:
                logger.info(f"關鍵字匹配足夠，返回 {len(final_candidates)} 筆候選")
            
            return final_candidates
            
        except Exception as e:
            logger.error(f"搜尋失敗: {e}")
            raise
    
    def _extract_keywords(self, instruction: str) -> List[str]:
        """從指令中提取關鍵字"""
        import re
        
        # 移除標點符號並轉為小寫
        cleaned = re.sub(r'[^\w\s]', ' ', instruction.lower())
        
        # 定義停用詞（常見但無意義的詞）
        stop_words = {
            '請', '到', '去', '走', '移動', '找', '找到', '尋找', '搜尋',
            '的', '在', '裡', '中', '上', '下', '左', '右', '前', '後',
            '那', '這', '個', '位置', '地方', '附近', '旁邊', '邊'
        }
        
        # 分詞並過濾停用詞
        words = [word.strip() for word in cleaned.split() if word.strip() and word.strip() not in stop_words]
        
        # 添加完整指令作為關鍵字（用於完整匹配）
        keywords = [instruction.lower().strip()]
        keywords.extend(words)
        
        return list(set(keywords))  # 去重
    
    def _calculate_keyword_match(self, ai_answer: str, keywords: List[str], full_instruction: str) -> float:
        """計算關鍵字匹配分數"""
        if not ai_answer or not keywords:
            return 0.0
        
        score = 0.0
        ai_answer_lower = ai_answer.lower()
        
        # 完整指令匹配（最高權重）
        if full_instruction in ai_answer_lower:
            score += 10.0
        
        # 關鍵字匹配
        for keyword in keywords:
            if keyword in ai_answer_lower:
                # 根據關鍵字長度給予不同權重
                if len(keyword) >= 3:
                    score += 3.0  # 長關鍵字權重較高
                else:
                    score += 1.0  # 短關鍵字權重較低
        
        # 模糊匹配（包含部分字符）
        for keyword in keywords:
            if len(keyword) >= 2:
                # 檢查是否包含關鍵字的部分
                for i in range(len(keyword) - 1):
                    if keyword[i:i+2] in ai_answer_lower:
                        score += 0.5
        
        return score
    
    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """解碼base64影像"""
        try:
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            return image
            
        except Exception as e:
            logger.error(f"影像解碼失敗: {e}")
            return None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """獲取Milvus集合資訊"""
        try:
            if self.collection:
                stats = self.collection.num_entities
                return {
                    "database_type": "Milvus",
                    "collection_name": self.collection_name,
                    "total_records": stats,
                    "status": "connected"
                }
            return {
                "database_type": "Milvus",
                "collection_name": self.collection_name,
                "total_records": 0,
                "status": "disconnected"
            }
        except Exception as e:
            return {
                "database_type": "Milvus",
                "collection_name": self.collection_name,
                "total_records": 0,
                "status": f"error: {str(e)}"
            }

class QdrantVisionSearcher(BaseVisionSearcher):
    def __init__(self, 
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333,
                 collection_name: str = "ros2_camera_image",
                 api_key: Optional[str] = None):
        """初始化Qdrant搜尋系統"""
        try:
            self.database_type = DatabaseType.QDRANT
            self.collection_name = collection_name
            self.client = None
            
            logger.info(f"初始化 Qdrant QdrantVisionSearcher")
            
            # 連接Qdrant
            self._connect_qdrant(qdrant_host, qdrant_port, api_key)
            
            # 檢查Collection
            self._check_collection()
            
            logger.info("Qdrant QdrantVisionSearcher 初始化完成")
            
        except Exception as e:
            logger.error(f"Qdrant QdrantVisionSearcher 初始化失敗: {e}")
            raise
    
    def _connect_qdrant(self, host: str, port: int, api_key: Optional[str] = None):
        """連接到Qdrant"""
        try:
            if api_key:
                self.client = QdrantClient(
                    host=host,
                    port=port,
                    api_key=api_key
                )
            else:
                self.client = QdrantClient(
                    host=host,
                    port=port
                )
            
            # 測試連接
            collections = self.client.get_collections()
            logger.info(f"成功連接到Qdrant: {host}:{port}")
            logger.info(f"可用的集合: {[c.name for c in collections.collections]}")
            
        except Exception as e:
            logger.error(f"連接Qdrant失敗: {e}")
            raise
    
    def _check_collection(self):
        """檢查Collection是否存在"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"找到Qdrant集合: {self.collection_name}")
            logger.info(f"集合包含 {collection_info.points_count} 個點")
            
        except Exception as e:
            logger.error(f"Qdrant集合檢查失敗: {e}")
            # 不拋出異常，讓用戶知道集合不存在但仍可使用其他功能
            logger.warning(f"集合 '{self.collection_name}' 可能不存在，請檢查集合名稱")
    
    def search_candidates(self, instruction: str, top_k: int = 50, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """根據指令搜尋候選位置 - 使用關鍵字篩選payload中的ai_answer欄位"""
        try:
            logger.info(f"正在從Qdrant搜尋指令: '{instruction}'，使用關鍵字篩選ai_answer欄位")
            
            # 提取指令中的關鍵字
            keywords = self._extract_keywords(instruction)
            logger.info(f"提取的關鍵字: {keywords}")
            
            # 設定搜尋限制，先獲取更多記錄用於篩選
            initial_limit = min(500, top_k * 10)
            
            # 從Qdrant搜尋所有點
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=initial_limit,
                with_payload=True
            )
            
            results = search_result[0]  # scroll返回 (points, next_page_offset)
            
            logger.info(f"從Qdrant獲取了 {len(results)} 筆記錄，目標需要 {top_k} 筆候選")
            
            # 根據關鍵字篩選候選
            matched_candidates = []  # 有關鍵字匹配的候選
            all_candidates = []      # 所有候選（包括無匹配的）
            
            for point in results:
                payload = point.payload
                ai_answer = payload.get('ai_answer', '').lower() if payload.get('ai_answer') else ''
                
                # 計算關鍵字匹配度
                match_score = self._calculate_keyword_match(ai_answer, keywords, instruction.lower())
                
                candidate = {
                    'id': str(point.id),
                    'keyword_match_score': match_score,  # 關鍵字匹配分數
                    'llava_score': None,  # 添加LLaVA分數欄位
                    'llava_reasoning': None,  # 添加LLaVA理由欄位
                    'llava_confidence': None,  # 添加LLaVA置信度欄位
                    'image_base64': payload.get('image_base64'),
                    'camera_path': payload.get('camera_path'),
                    'ai_answer': payload.get('ai_answer'),
                    'ai_question': payload.get('ai_question'),
                    'position': {
                        'x': float(payload.get('position_x', 0)),
                        'y': float(payload.get('position_y', 0)),
                        'z': float(payload.get('position_z', 0))
                    },
                    'rotation': {
                        'x': float(payload.get('rotation_x', 0)),
                        'y': float(payload.get('rotation_y', 0)),
                        'z': float(payload.get('rotation_z', 0)),
                        'w': float(payload.get('rotation_w', 1))
                    },
                    'metadata': payload.get('metadata'),
                    'timestamp': payload.get('timestamp')
                }
                
                if match_score > 0:
                    matched_candidates.append(candidate)
                else:
                    all_candidates.append(candidate)
            
            # 先排序匹配的候選
            matched_candidates.sort(key=lambda x: x['keyword_match_score'], reverse=True)
            
            # 組合最終結果：優先使用匹配的候選，不足時補充其他候選
            final_candidates = []
            
            # 添加匹配的候選
            final_candidates.extend(matched_candidates[:top_k])
            
            # 如果匹配的候選不夠，補充其他候選
            if len(final_candidates) < top_k:
                additional_needed = top_k - len(final_candidates)
                
                # 從未匹配的候選中補充（隨機打亂以增加多樣性）
                import random
                random.shuffle(all_candidates)
                
                # 確保不重複添加
                existing_ids = {c['id'] for c in final_candidates}
                for candidate in all_candidates:
                    if candidate['id'] not in existing_ids and len(final_candidates) < top_k:
                        final_candidates.append(candidate)
                
                logger.info(f"關鍵字匹配 {len(matched_candidates)} 筆，補充 {len(final_candidates) - len(matched_candidates)} 筆，共 {len(final_candidates)} 筆")
            else:
                logger.info(f"關鍵字匹配足夠，返回 {len(final_candidates)} 筆候選")
            
            return final_candidates
            
        except Exception as e:
            logger.error(f"Qdrant搜尋失敗: {e}")
            raise
    
    def _extract_keywords(self, instruction: str) -> List[str]:
        """從指令中提取關鍵字 - 與Milvus版本相同"""
        import re
        
        # 移除標點符號並轉為小寫
        cleaned = re.sub(r'[^\w\s]', ' ', instruction.lower())
        
        # 定義停用詞（常見但無意義的詞）
        stop_words = {
            '請', '到', '去', '走', '移動', '找', '找到', '尋找', '搜尋',
            '的', '在', '裡', '中', '上', '下', '左', '右', '前', '後',
            '那', '這', '個', '位置', '地方', '附近', '旁邊', '邊'
        }
        
        # 分詞並過濾停用詞
        words = [word.strip() for word in cleaned.split() if word.strip() and word.strip() not in stop_words]
        
        # 添加完整指令作為關鍵字（用於完整匹配）
        keywords = [instruction.lower().strip()]
        keywords.extend(words)
        
        return list(set(keywords))  # 去重
    
    def _calculate_keyword_match(self, ai_answer: str, keywords: List[str], full_instruction: str) -> float:
        """計算關鍵字匹配分數 - 與Milvus版本相同"""
        if not ai_answer or not keywords:
            return 0.0
        
        score = 0.0
        ai_answer_lower = ai_answer.lower()
        
        # 完整指令匹配（最高權重）
        if full_instruction in ai_answer_lower:
            score += 10.0
        
        # 關鍵字匹配
        for keyword in keywords:
            if keyword in ai_answer_lower:
                # 根據關鍵字長度給予不同權重
                if len(keyword) >= 3:
                    score += 3.0  # 長關鍵字權重較高
                else:
                    score += 1.0  # 短關鍵字權重較低
        
        # 模糊匹配（包含部分字符）
        for keyword in keywords:
            if len(keyword) >= 2:
                # 檢查是否包含關鍵字的部分
                for i in range(len(keyword) - 1):
                    if keyword[i:i+2] in ai_answer_lower:
                        score += 0.5
        
        return score
    
    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """解碼base64影像 - 與Milvus版本相同"""
        try:
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            return image
            
        except Exception as e:
            logger.error(f"影像解碼失敗: {e}")
            return None
    
    def get_collection_info(self) -> Dict[str, Any]:
        """獲取Qdrant集合資訊"""
        try:
            if self.client:
                collection_info = self.client.get_collection(self.collection_name)
                return {
                    "database_type": "Qdrant",
                    "collection_name": self.collection_name,
                    "total_records": collection_info.points_count,
                    "status": "connected"
                }
            return {
                "database_type": "Qdrant",
                "collection_name": self.collection_name,
                "total_records": 0,
                "status": "disconnected"
            }
        except Exception as e:
            return {
                "database_type": "Qdrant",
                "collection_name": self.collection_name,
                "total_records": 0,
                "status": f"error: {str(e)}"
            }

def create_vision_searcher(database_type: DatabaseType, **kwargs) -> BaseVisionSearcher:
    """工廠函數：根據資料庫類型創建對應的搜尋器"""
    if database_type == DatabaseType.MILVUS:
        if not QDRANT_AVAILABLE:  # 檢查是否可用 - 修正邏輯
            logger.warning("注意：僅支援 Milvus 資料庫")
        return RobotVisionSearcher(
            milvus_host=kwargs.get('host', 'localhost'),
            milvus_port=kwargs.get('port', '19530'),
            collection_name=kwargs.get('collection_name', 'ros2_camera_images')
        )
    elif database_type == DatabaseType.QDRANT:
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant客戶端不可用，請安裝 qdrant-client: pip install qdrant-client")
        return QdrantVisionSearcher(
            qdrant_host=kwargs.get('host', 'localhost'),
            qdrant_port=int(kwargs.get('port', 6333)),
            collection_name=kwargs.get('collection_name', 'ros2_camera_images'),  # 修正為複數形式
            api_key=kwargs.get('api_key', None)
        )
    else:
        raise ValueError(f"不支援的資料庫類型: {database_type}")

class SafeRobotVisionSearchGUI:
    def __init__(self):
        try:
            logger.info("開始創建GUI...")
            
            # 基本Tkinter設置
            self.root = tk.Tk()
            self.root.withdraw()  # 暫時隱藏窗口，直到完全初始化
            
            self.root.title("機器人視覺記錄搜尋系統 - 純語義搜尋 + ROS Bridge")
            self.root.geometry("1800x1100")  # 增加視窗大小以容納更多內容
            
            # 初始化變數
            self.searcher = None
            self.llava_client = None
            self.isaac_client = None
            self.current_candidates = []
            self.current_instruction = ""
            self.llava_evaluation_progress = 0
            self.llava_evaluation_total = 0
            self.selected_candidate = None
            
            # 資料庫選擇變數
            self.database_type = DatabaseType.MILVUS  # 預設使用 Milvus
            self.collection_name = ""  # 將在 GUI 中設定
            
            logger.info("基本變數初始化完成")
            
            # 創建GUI（分步驟進行）
            self.create_widgets()
            
            logger.info("GUI元件創建完成")
            
            # 啟動定時器檢查Isaac Sim狀態
            self.update_isaac_status_timer()
            
            # 顯示窗口
            self.root.deiconify()
            
            logger.info("GUI 創建完成，請點擊'重新連接'按鈕來初始化系統")
            
        except Exception as e:
            logger.error(f"GUI 初始化失敗: {e}")
            traceback.print_exc()
            # 確保窗口關閉
            if hasattr(self, 'root'):
                try:
                    self.root.destroy()
                except:
                    pass
            raise
    
    def create_widgets(self):
        """創建GUI元件 - 分步驟安全版本"""
        try:
            logger.info("創建主框架...")
            # 主框架
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # 配置網格權重 - 讓搜尋結果佔一半界面
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(1, weight=1)
            # 為上半部分配置少量權重
            main_frame.rowconfigure(0, weight=1)  # 連接設定
            main_frame.rowconfigure(1, weight=1)  # 搜尋設定
            main_frame.rowconfigure(2, weight=0)  # 常用指令 (固定高度)
            main_frame.rowconfigure(3, weight=1)  # Isaac導航
            main_frame.rowconfigure(4, weight=6)  # 搜尋結果 (佔最大權重)
            
            logger.info("創建連接設定區域...")
            self.create_connection_frame(main_frame)
            
            logger.info("創建搜尋設定區域...")
            self.create_search_frame(main_frame)
            
            logger.info("創建常用指令區域...")
            self.create_shortcuts_frame(main_frame)
            
            logger.info("創建Isaac Sim控制區域...")
            self.create_isaac_frame(main_frame)
            
            logger.info("創建結果顯示區域...")
            self.create_results_frame(main_frame)
            
            logger.info("創建進度條...")
            self.create_progress_bars(main_frame)
            
            logger.info("所有GUI元件創建完成")
            
        except Exception as e:
            logger.error(f"GUI元件創建失敗: {e}")
            raise
    
    def create_connection_frame(self, parent):
        """創建連接設定區域"""
        connection_frame = ttk.LabelFrame(parent, text="連接設定", padding="5")
        connection_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # 資料庫類型選擇
        db_type_frame = ttk.Frame(connection_frame)
        db_type_frame.grid(row=0, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(db_type_frame, text="資料庫類型:").grid(row=0, column=0, sticky=tk.W)
        self.db_type_var = tk.StringVar(value="milvus")
        ttk.Radiobutton(db_type_frame, text="Milvus", variable=self.db_type_var, value="milvus", 
                       command=self.on_database_type_change).grid(row=0, column=1, padx=(5, 10))
        if QDRANT_AVAILABLE:
            ttk.Radiobutton(db_type_frame, text="Qdrant", variable=self.db_type_var, value="qdrant",
                           command=self.on_database_type_change).grid(row=0, column=2, padx=(5, 10))
        else:
            ttk.Label(db_type_frame, text="Qdrant (不可用)", foreground="gray").grid(row=0, column=2, padx=(5, 10))
        
        # Collection 名稱設定
        collection_frame = ttk.Frame(connection_frame)
        collection_frame.grid(row=1, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(5, 10))
        
        ttk.Label(collection_frame, text="Collection:").grid(row=0, column=0, sticky=tk.W)
        self.collection_entry = ttk.Entry(collection_frame, width=40)
        self.collection_entry.insert(0, "ros2_camera_images")  # Milvus 預設
        self.collection_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        collection_frame.columnconfigure(1, weight=1)
        
        # 資料庫連接設定
        db_conn_frame = ttk.Frame(connection_frame)
        db_conn_frame.grid(row=2, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(5, 10))
        
        ttk.Label(db_conn_frame, text="Host:").grid(row=0, column=0, sticky=tk.W)
        self.host_entry = ttk.Entry(db_conn_frame, width=15)
        self.host_entry.insert(0, "localhost")
        self.host_entry.grid(row=0, column=1, padx=(5, 10))
        
        ttk.Label(db_conn_frame, text="Port:").grid(row=0, column=2, sticky=tk.W)
        self.port_entry = ttk.Entry(db_conn_frame, width=10)
        self.port_entry.insert(0, "19530")  # Milvus 預設
        self.port_entry.grid(row=0, column=3, padx=(5, 10))
        
        # Qdrant API Key (可選)
        ttk.Label(db_conn_frame, text="API Key:").grid(row=0, column=4, sticky=tk.W)
        self.api_key_entry = ttk.Entry(db_conn_frame, width=15, show="*")
        self.api_key_entry.grid(row=0, column=5, padx=(5, 10))
        self.api_key_entry.grid_remove()  # 初始隱藏，只在選擇 Qdrant 時顯示
        
        self.connect_btn = ttk.Button(db_conn_frame, text="重新連接", command=self.initialize_connections)
        self.connect_btn.grid(row=0, column=6, padx=(10, 5))
        
        self.db_status_label = ttk.Label(db_conn_frame, text="未連接", foreground="red")
        self.db_status_label.grid(row=0, column=7, padx=(5, 0))
        
        # LLaVA設定
        llava_frame = ttk.Frame(connection_frame)
        llava_frame.grid(row=3, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(5, 10))
        
        ttk.Label(llava_frame, text="LLaVA API:").grid(row=0, column=0, sticky=tk.W)
        self.llava_url_entry = ttk.Entry(llava_frame, width=25)
        self.llava_url_entry.insert(0, "http://localhost:11434")
        self.llava_url_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(5, 10))
        
        ttk.Label(llava_frame, text="模型:").grid(row=0, column=3, sticky=tk.W)
        self.llava_model_entry = ttk.Entry(llava_frame, width=10)
        self.llava_model_entry.insert(0, "llava")
        self.llava_model_entry.grid(row=0, column=4, padx=(5, 10))
        
        self.llava_status_label = ttk.Label(llava_frame, text="未連接", foreground="red")
        self.llava_status_label.grid(row=0, column=5, padx=(10, 0))
        llava_frame.columnconfigure(1, weight=1)
        
        # ROS Bridge 設定
        ros_frame = ttk.Frame(connection_frame)
        ros_frame.grid(row=4, column=0, columnspan=6, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(ros_frame, text="ROS Bridge:").grid(row=0, column=0, sticky=tk.W)
        self.isaac_url_entry = ttk.Entry(ros_frame, width=25)
        self.isaac_url_entry.insert(0, "ws://localhost:9090")
        self.isaac_url_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=(5, 10))
        
        ttk.Label(ros_frame, text="機器人:").grid(row=0, column=3, sticky=tk.W)
        self.robot_name_entry = ttk.Entry(ros_frame, width=10)
        self.robot_name_entry.insert(0, "Baymax")
        self.robot_name_entry.grid(row=0, column=4, padx=(5, 10))
        
        self.isaac_status_label = ttk.Label(ros_frame, text="未連接", foreground="red")
        self.isaac_status_label.grid(row=0, column=5, padx=(10, 0))
        ros_frame.columnconfigure(1, weight=1)
    
    def on_database_type_change(self):
        """資料庫類型切換回調"""
        db_type = self.db_type_var.get()
        
        if db_type == "milvus":
            self.database_type = DatabaseType.MILVUS
            # 更新預設值
            self.collection_entry.delete(0, tk.END)
            self.collection_entry.insert(0, "ros2_camera_images")
            self.port_entry.delete(0, tk.END)
            self.port_entry.insert(0, "19530")
            # 隱藏 API Key
            self.api_key_entry.grid_remove()
            logger.info("切換到 Milvus 資料庫模式")
            
        elif db_type == "qdrant":
            self.database_type = DatabaseType.QDRANT
            # 更新預設值
            self.collection_entry.delete(0, tk.END)
            self.collection_entry.insert(0, "ros2_camera_images")  # 修正為複數形式
            self.port_entry.delete(0, tk.END)
            self.port_entry.insert(0, "6333")
            # 顯示 API Key
            self.api_key_entry.grid()
            logger.info("切換到 Qdrant 資料庫模式")
        
        # 重置連接狀態
        self.db_status_label.configure(text="未連接", foreground="red")
        if hasattr(self, 'searcher'):
            self.searcher = None
    
    def create_search_frame(self, parent):
        """創建搜尋設定區域"""
        # 搜尋區域
        search_frame = ttk.LabelFrame(parent, text="搜尋設定", padding="5")
        search_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        search_frame.columnconfigure(1, weight=1)
        
        ttk.Label(search_frame, text="指令:").grid(row=0, column=0, sticky=tk.W)
        self.instruction_entry = ttk.Entry(search_frame, width=50)
        self.instruction_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 10))
        self.instruction_entry.bind('<Return>', lambda e: self.search_instruction())
        
        # 搜尋參數
        params_frame = ttk.Frame(search_frame)
        params_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(params_frame, text="候選數量:").grid(row=0, column=0, sticky=tk.W)
        self.top_k_var = tk.StringVar(value="20")
        ttk.Entry(params_frame, textvariable=self.top_k_var, width=5).grid(row=0, column=1, padx=(5, 20))
        
        ttk.Label(params_frame, text="資料來源:").grid(row=0, column=2, sticky=tk.W)
        self.db_info_label = ttk.Label(params_frame, text="請先連接資料庫", foreground="gray")
        self.db_info_label.grid(row=0, column=3, padx=(5, 20))
        
        self.search_btn = ttk.Button(params_frame, text="🔍 獲取候選", command=self.search_instruction, state=tk.DISABLED)
        self.search_btn.grid(row=0, column=4, padx=(20, 10))
        
        self.llava_btn = ttk.Button(params_frame, text="🤖 LLaVA視覺+語義評分", command=self.evaluate_with_llava, state=tk.DISABLED)
        self.llava_btn.grid(row=0, column=5, padx=(5, 10))
        
        self.export_btn = ttk.Button(params_frame, text="💾 匯出結果", command=self.export_results, state=tk.DISABLED)
        self.export_btn.grid(row=0, column=6, padx=(5, 0))
        
        # 排序選項
        sort_frame = ttk.Frame(search_frame)
        sort_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(sort_frame, text="排序依據:").grid(row=0, column=0, sticky=tk.W)
        self.sort_var = tk.StringVar(value="llava")  # 預設使用LLaVA排序
        ttk.Radiobutton(sort_frame, text="原始順序", variable=self.sort_var, value="original", command=self.resort_candidates).grid(row=0, column=1, padx=(5, 10))
        ttk.Radiobutton(sort_frame, text="LLaVA視覺+語義分數", variable=self.sort_var, value="llava", command=self.resort_candidates).grid(row=0, column=2, padx=(5, 10))
        ttk.Radiobutton(sort_frame, text="時間順序", variable=self.sort_var, value="time", command=self.resort_candidates).grid(row=0, column=3, padx=(5, 10))
    
    def create_shortcuts_frame(self, parent):
        """創建常用指令區域"""
        # 常用指令按鈕
        shortcuts_frame = ttk.LabelFrame(parent, text="🔥 常用指令", padding="5")
        shortcuts_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(2, 2))
        
        # 常用指令（只保留三個主要指令）
        shortcuts = [
            "請走到三角錐的位置",
            "請走到桌子那",
            "走到架子旁邊"
        ]
        
        # 創建按鈕網格，一行顯示3個按鈕
        for i, shortcut in enumerate(shortcuts):
            btn = ttk.Button(shortcuts_frame, text=shortcut, 
                           command=lambda s=shortcut: self.set_instruction(s),
                           width=16)
            btn.grid(row=0, column=i, padx=4, pady=3, sticky=(tk.W, tk.E))
        
        # 配置列權重，讓按鈕平均分佈
        for i in range(3):
            shortcuts_frame.columnconfigure(i, weight=1)
        
        # 添加資料庫資訊提示
        info_frame = ttk.Frame(shortcuts_frame)
        info_frame.grid(row=1, column=0, columnspan=3, pady=(5, 0))
        
        info_text = "💡 提示：指令將在所選資料庫中搜尋相關影像和位置資訊"
        ttk.Label(info_frame, text=info_text, foreground="gray", font=("TkDefaultFont", 7)).pack()
    
    def create_isaac_frame(self, parent):
        """創建ROS Bridge機器人控制區域"""
        # ROS Bridge 控制區域
        isaac_frame = ttk.LabelFrame(parent, text="ROS Bridge 機器人控制 (透過 ros2_navigation extension)", padding="5")
        isaac_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        isaac_frame.columnconfigure(1, weight=1)
        
        # 導航控制
        nav_control_frame = ttk.Frame(isaac_frame)
        nav_control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.navigate_btn = ttk.Button(nav_control_frame, text="🚀 導航到選中位置", command=self.navigate_to_selected, state=tk.DISABLED)
        self.navigate_btn.grid(row=0, column=0, padx=(0, 10))
        
        self.auto_nav_var = tk.BooleanVar()
        self.auto_nav_check = ttk.Checkbutton(nav_control_frame, text="LLaVA評分後自動導航", variable=self.auto_nav_var)
        self.auto_nav_check.grid(row=0, column=1, padx=(0, 10))
        
        self.isaac_status_btn = ttk.Button(nav_control_frame, text="📊 檢查狀態", command=self.check_isaac_status, state=tk.DISABLED)
        self.isaac_status_btn.grid(row=0, column=2, padx=(0, 10))
        
        # 導航狀態顯示
        self.nav_status_label = ttk.Label(isaac_frame, text="請選擇一個候選位置進行導航", foreground="gray")
        self.nav_status_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
    
    def create_results_frame(self, parent):
        """創建結果顯示區域"""
        # 結果顯示區域
        results_frame = ttk.LabelFrame(parent, text="搜尋結果", padding="5")
        results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=2)  # 候選列表佔較小比例
        results_frame.columnconfigure(1, weight=3)  # 詳細資訊佔較大比例
        results_frame.rowconfigure(1, weight=1)
        
        # 左側：候選列表
        left_frame = ttk.Frame(results_frame)
        left_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        left_frame.columnconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=1)
        
        ttk.Label(left_frame, text="候選位置列表").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # 創建候選列表 - 移除CLIP分數欄，調整欄位寬度
        columns = ('排名', 'LLaVA分數', '座標', 'AI描述')
        self.candidates_tree = ttk.Treeview(left_frame, columns=columns, show='headings', height=18)
        
        self.candidates_tree.heading('排名', text='排名')
        self.candidates_tree.heading('LLaVA分數', text='LLaVA分數')
        self.candidates_tree.heading('座標', text='座標 (x, y, z)')
        self.candidates_tree.heading('AI描述', text='AI描述')
        
        self.candidates_tree.column('排名', width=40)
        self.candidates_tree.column('LLaVA分數', width=80)
        self.candidates_tree.column('座標', width=100)
        self.candidates_tree.column('AI描述', width=150)
        
        # 添加滾動條
        candidates_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.candidates_tree.yview)
        self.candidates_tree.configure(yscrollcommand=candidates_scrollbar.set)
        
        self.candidates_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        candidates_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # 綁定選擇事件
        self.candidates_tree.bind('<<TreeviewSelect>>', self.on_candidate_select)
        
        # 調用輔助方法創建剩餘的GUI元件
        self.create_widgets_continued(parent, results_frame)
    
    def create_progress_bars(self, parent):
        """創建進度條"""
        # 進度條
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # LLaVA評分進度條
        self.llava_progress = ttk.Progressbar(parent, mode='determinate')
        self.llava_progress.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        self.llava_progress.grid_remove()  # 初始隱藏
        
        self.llava_progress_label = ttk.Label(parent, text="")
        self.llava_progress_label.grid(row=7, column=0, columnspan=2, pady=(2, 0))
    
    def update_isaac_status_timer(self):
        """定期更新ROS Bridge狀態"""
        # 每5秒檢查一次ROS Bridge連接狀態
        if self.isaac_client:
            try:
                # 檢查連接狀態並更新UI
                if hasattr(self.isaac_client, 'connected'):
                    if self.isaac_client.connected:
                        if hasattr(self, 'isaac_status_label'):
                            self.isaac_status_label.configure(text="已連接", foreground="green")
                    else:
                        if hasattr(self, 'isaac_status_label'):
                            self.isaac_status_label.configure(text="連接中斷", foreground="orange")
            except:
                pass
        
        # 每5秒更新一次
        self.root.after(5000, self.update_isaac_status_timer)
    
    def navigate_to_selected(self):
        """導航到選中的候選位置"""
        if not self.isaac_client:
            messagebox.showerror("錯誤", "ROS Bridge未連接")
            return
        
        if not self.selected_candidate:
            messagebox.showwarning("警告", "請先選擇一個候選位置")
            return
        
        def navigate_worker():
            try:
                self.root.after(0, lambda: self.navigate_btn.configure(state=tk.DISABLED))
                self.root.after(0, lambda: self.nav_status_label.configure(text="正在發送導航指令...", foreground="blue"))
                
                position = self.selected_candidate['position']
                result = self.isaac_client.navigate_to_position(position)
                
                if result['success']:
                    success_msg = f"✅ 導航指令已發送到位置: ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})"
                    self.root.after(0, lambda: self.nav_status_label.configure(text=success_msg, foreground="green"))
                    logger.info(f"導航成功: {success_msg}")
                else:
                    error_msg = f"❌ 導航失敗: {result.get('error', '未知錯誤')}"
                    self.root.after(0, lambda: self.nav_status_label.configure(text=error_msg, foreground="red"))
                    self.root.after(0, lambda: messagebox.showerror("導航失敗", result.get('error', '未知錯誤')))
                
            except Exception as e:
                error_msg = f"導航過程中發生錯誤: {str(e)}"
                logger.error(error_msg)
                self.root.after(0, lambda: self.nav_status_label.configure(text=f"❌ {error_msg}", foreground="red"))
                self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
            finally:
                self.root.after(0, lambda: self.navigate_btn.configure(state=tk.NORMAL))
        
        thread = threading.Thread(target=navigate_worker, daemon=True)
        thread.start()
    
    def check_isaac_status(self):
        """檢查ROS Bridge狀態"""
        if not self.isaac_client:
            messagebox.showerror("錯誤", "ROS Bridge未連接")
            return
        
        def status_worker():
            try:
                status = self.isaac_client.get_status()
                
                if status.get('success', False):
                    status_info = f"✅ ROS Bridge狀態正常\n"
                    status_info += f"機器人: {status.get('robot_name', 'N/A')}\n"
                    status_info += f"連接狀態: {'已連接' if status.get('connected', False) else '未連接'}\n"
                    status_info += f"導航狀態: {status.get('navigation_status', 'N/A')}\n"
                    status_info += f"檢查時間: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status.get('timestamp', time.time())))}"
                    
                    self.root.after(0, lambda: messagebox.showinfo("ROS Bridge狀態", status_info))
                else:
                    error_msg = f"❌ 狀態檢查失敗: {status.get('error', '未知錯誤')}"
                    self.root.after(0, lambda: messagebox.showerror("狀態錯誤", error_msg))
                    
            except Exception as e:
                error_msg = f"狀態檢查過程中發生錯誤: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
        
        thread = threading.Thread(target=status_worker, daemon=True)
        thread.start()
    
    def auto_navigate_best_candidate(self):
        """自動導航到最佳候選位置（LLaVA評分最高）"""
        if not self.auto_nav_var.get():
            return
        
        if not self.isaac_client:
            logger.warning("自動導航失敗：ROS Bridge未連接")
            return
        
        # 找到LLaVA分數最高的候選
        best_candidate = None
        best_score = -1
        
        for candidate in self.current_candidates:
            if candidate['llava_score'] is not None and candidate['llava_score'] > best_score:
                best_candidate = candidate
                best_score = candidate['llava_score']
        
        if best_candidate and best_score >= 60:  # 只有評分60以上才自動導航
            self.selected_candidate = best_candidate
            
            # 在樹狀列表中選中該候選
            for item in self.candidates_tree.get_children():
                values = self.candidates_tree.item(item)['values']
                if values and len(values) > 2 and str(values[2]) == str(best_score):
                    self.candidates_tree.selection_set(item)
                    self.candidates_tree.see(item)
                    break
            
            logger.info(f"自動導航到最佳候選位置，LLaVA分數: {best_score}")
            self.navigate_to_selected()
        else:
            logger.info(f"最佳候選分數過低({best_score})，跳過自動導航")
    
    def create_widgets_continued(self, main_frame, results_frame):
        """繼續創建GUI元件 - 右側詳細資訊區域"""
        # 右側：詳細資訊
        right_frame = ttk.Frame(results_frame)
        right_frame.grid(row=0, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(2, weight=1)
        
        ttk.Label(right_frame, text="詳細資訊").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # 影像顯示區域 - 縮小高度，給文字區域更多空間
        self.image_frame = ttk.Frame(right_frame, relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.image_frame.grid_propagate(False)
        self.image_frame.configure(height=150, width=300)  # 降低高度從200到150
        
        self.image_label = ttk.Label(self.image_frame, text="選擇候選位置以顯示影像")
        self.image_label.pack(expand=True)
        
        # 詳細資訊文本區域 - 增加高度
        self.detail_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, height=20)  # 增加高度從15到20
        self.detail_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def set_instruction(self, instruction):
        """設定指令到輸入框"""
        self.instruction_entry.delete(0, tk.END)
        self.instruction_entry.insert(0, instruction)
    
    def initialize_connections(self):
        """初始化連接"""
        def init_worker():
            try:
                self.root.after(0, lambda: self.progress.start())
                
                # 獲取連接參數
                host = self.host_entry.get()
                port = self.port_entry.get()
                collection_name = self.collection_entry.get()
                api_key = self.api_key_entry.get() if self.api_key_entry.get() else None
                
                logger.info(f"嘗試連接 {self.database_type.value.upper()}: {host}:{port}, Collection: {collection_name}")
                
                # 初始化資料庫連接
                try:
                    if self.database_type == DatabaseType.MILVUS:
                        self.searcher = create_vision_searcher(
                            DatabaseType.MILVUS,
                            host=host,
                            port=port,
                            collection_name=collection_name
                        )
                    elif self.database_type == DatabaseType.QDRANT:
                        self.searcher = create_vision_searcher(
                            DatabaseType.QDRANT,
                            host=host,
                            port=int(port),
                            collection_name=collection_name,
                            api_key=api_key
                        )
                    
                    # 獲取並顯示資料庫資訊
                    db_info = self.searcher.get_collection_info()
                    status_text = f"已連接 - {db_info['total_records']} 筆記錄"
                    db_source_text = f"{db_info['database_type']} - {db_info['collection_name']}"
                    self.root.after(0, lambda: self.db_status_label.configure(text=status_text, foreground="green"))
                    self.root.after(0, lambda: self.db_info_label.configure(text=db_source_text, foreground="blue"))
                    self.root.after(0, lambda: self.search_btn.configure(state=tk.NORMAL))
                    logger.info(f"{self.database_type.value.upper()} 連接成功: {db_info}")
                    
                except Exception as db_error:
                    logger.error(f"{self.database_type.value.upper()} 連接失敗: {db_error}")
                    self.root.after(0, lambda: self.db_status_label.configure(text="連接失敗", foreground="red"))
                    # 不要因為資料庫連接失敗就終止，繼續嘗試其他服務
                    self.searcher = None
                
                # 初始化LLaVA連接
                llava_url = self.llava_url_entry.get()
                llava_model = self.llava_model_entry.get()
                
                logger.info(f"嘗試連接 LLaVA: {llava_url}")
                
                try:
                    self.llava_client = OllamaLLaVAClient(
                        base_url=llava_url,
                        model_name=llava_model
                    )
                    
                    if self.llava_client.test_connection():
                        self.root.after(0, lambda: self.llava_status_label.configure(text="已連接", foreground="green"))
                        logger.info("LLaVA 連接成功")
                        # 只有在有搜尋器的情況下才啟用 LLaVA 按鈕
                        if self.searcher:
                            self.root.after(0, lambda: self.llava_btn.configure(state=tk.NORMAL))
                    else:
                        self.root.after(0, lambda: self.llava_status_label.configure(text="連接失敗", foreground="red"))
                        logger.warning("LLaVA 連接失敗")
                        
                except Exception as llava_error:
                    logger.error(f"LLaVA 連接失敗: {llava_error}")
                    self.root.after(0, lambda: self.llava_status_label.configure(text="連接失敗", foreground="red"))
                
                # 初始化ROS Bridge連接
                isaac_url = self.isaac_url_entry.get()
                
                logger.info(f"嘗試連接 ROS Bridge: {isaac_url}")
                
                try:
                    self.isaac_client = IsaacSimClient(base_url=isaac_url)
                    
                    if self.isaac_client.test_connection():
                        # 建立連接
                        if self.isaac_client.connect():
                            self.root.after(0, lambda: self.isaac_status_label.configure(text="已連接", foreground="green"))
                            self.root.after(0, lambda: self.navigate_btn.configure(state=tk.NORMAL))
                            self.root.after(0, lambda: self.isaac_status_btn.configure(state=tk.NORMAL))
                            logger.info("ROS Bridge 連接成功")
                        else:
                            self.root.after(0, lambda: self.isaac_status_label.configure(text="連接失敗", foreground="red"))
                            logger.warning("ROS Bridge 連接失敗")
                    else:
                        self.root.after(0, lambda: self.isaac_status_label.configure(text="服務不可用", foreground="red"))
                        logger.warning("ROS Bridge 服務不可用")
                        
                except Exception as isaac_error:
                    logger.error(f"ROS Bridge 連接失敗: {isaac_error}")
                    self.root.after(0, lambda: self.isaac_status_label.configure(text="連接失敗", foreground="red"))
                
            except Exception as e:
                error_msg = f"初始化過程中發生錯誤: {str(e)}"
                logger.error(error_msg)
                self.root.after(0, lambda: self.milvus_status_label.configure(text="連接失敗", foreground="red"))
                self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
            finally:
                self.root.after(0, lambda: self.progress.stop())
        
        thread = threading.Thread(target=init_worker, daemon=True)
        thread.start()
    
    def search_instruction(self):
        """執行資料庫查詢獲取候選位置"""
        if not self.searcher:
            messagebox.showerror("錯誤", "請先連接到Milvus資料庫")
            return
        
        instruction = self.instruction_entry.get().strip()
        if not instruction:
            messagebox.showwarning("警告", "請輸入搜尋指令")
            return
        
        try:
            top_k = int(self.top_k_var.get())
        except ValueError:
            messagebox.showerror("錯誤", "請輸入有效的候選數量")
            return
        
        def search_worker():
            try:
                self.root.after(0, self.progress.start)
                self.root.after(0, lambda: self.search_btn.configure(state=tk.DISABLED))
                
                candidates = self.searcher.search_candidates(instruction, top_k, 0.0)
                
                self.current_instruction = instruction
                self.root.after(0, lambda: self.update_results(candidates, instruction))
                
            except Exception as e:
                error_msg = f"搜尋失敗: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
            finally:
                self.root.after(0, self.progress.stop)
                self.root.after(0, lambda: self.search_btn.configure(state=tk.NORMAL))
        
        thread = threading.Thread(target=search_worker, daemon=True)
        thread.start()
    
    def evaluate_with_llava(self):
        """使用LLaVA評估候選位置"""
        if not self.llava_client:
            messagebox.showerror("錯誤", "LLaVA未連接")
            return
        
        if not self.current_candidates:
            messagebox.showwarning("警告", "沒有候選位置可評估")
            return
        
        def evaluate_worker():
            try:
                self.root.after(0, lambda: self.llava_btn.configure(state=tk.DISABLED))
                self.root.after(0, lambda: self.llava_progress.grid())
                
                self.llava_evaluation_total = len(self.current_candidates)
                self.llava_evaluation_progress = 0
                
                self.root.after(0, lambda: self.llava_progress.configure(maximum=self.llava_evaluation_total, value=0))
                
                for i, candidate in enumerate(self.current_candidates):
                    self.root.after(0, lambda i=i: self.llava_progress_label.configure(
                        text=f"正在評估候選位置 {i+1}/{self.llava_evaluation_total}..."
                    ))
                    
                    # 使用LLaVA評估
                    result = self.llava_client.evaluate_candidate(
                        instruction=self.current_instruction,
                        image_base64=candidate['image_base64'],
                        ai_description=candidate.get('ai_answer', ''),
                        position=candidate['position']
                    )
                    
                    # 更新候選結果
                    candidate['llava_score'] = result['score']
                    candidate['llava_reasoning'] = result['reasoning']
                    candidate['llava_confidence'] = result['confidence']
                    
                    self.llava_evaluation_progress += 1
                    self.root.after(0, lambda: self.llava_progress.configure(value=self.llava_evaluation_progress))
                
                # 重新排序並更新顯示
                self.sort_var.set("llava")  # 自動切換到LLaVA排序
                self.root.after(0, self.resort_candidates)
                
                # 檢查是否需要自動導航
                self.root.after(0, self.auto_navigate_best_candidate)
                
            except Exception as e:
                error_msg = f"LLaVA評估失敗: {str(e)}"
                self.root.after(0, lambda: messagebox.showerror("錯誤", error_msg))
            finally:
                self.root.after(0, lambda: self.llava_btn.configure(state=tk.NORMAL))
                self.root.after(0, lambda: self.llava_progress.grid_remove())
                self.root.after(0, lambda: self.llava_progress_label.configure(text=""))
        
        thread = threading.Thread(target=evaluate_worker, daemon=True)
        thread.start()
    
    def resort_candidates(self):
        """重新排序候選結果"""
        if not self.current_candidates:
            return
        
        sort_by = self.sort_var.get()
        
        if sort_by == "original":
            # 按原始順序（資料庫返回順序）
            self.current_candidates = self.current_candidates
        elif sort_by == "llava":
            # 只顯示有LLaVA分數的候選，按分數排序
            candidates_with_llava = [c for c in self.current_candidates if c['llava_score'] is not None]
            candidates_without_llava = [c for c in self.current_candidates if c['llava_score'] is None]
            candidates_with_llava.sort(key=lambda x: x['llava_score'], reverse=True)
            self.current_candidates = candidates_with_llava + candidates_without_llava
        elif sort_by == "time":
            # 按時間戳排序（最新的在前）
            self.current_candidates.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        self.update_candidates_display()
    
    def update_results(self, candidates, instruction):
        """更新搜尋結果顯示"""
        self.current_candidates = candidates
        self.update_candidates_display()
        
        # 啟用LLaVA評分和匯出按鈕
        if self.llava_client and self.llava_client.test_connection():
            self.llava_btn.configure(state=tk.NORMAL)
        self.export_btn.configure(state=tk.NORMAL)
        
        # 清空詳細資訊
        self.detail_text.delete(1.0, tk.END)
        self.detail_text.insert(tk.END, f"搜尋指令: {instruction}\n")
        
        # 顯示資料庫資訊
        if self.searcher:
            db_info = self.searcher.get_collection_info()
            self.detail_text.insert(tk.END, f"資料來源: {db_info['database_type']} - {db_info['collection_name']}\n")
            self.detail_text.insert(tk.END, f"資料庫總記錄數: {db_info['total_records']}\n")
        
        self.detail_text.insert(tk.END, f"關鍵字篩選後獲取到 {len(candidates)} 個候選位置\n\n")
        self.detail_text.insert(tk.END, "請從左側列表選擇候選位置查看詳細資訊\n\n")
        self.detail_text.insert(tk.END, "💡 評估流程說明：\n")
        self.detail_text.insert(tk.END, "1. 系統根據關鍵字從ai_answer欄位中篩選相關記錄\n")
        self.detail_text.insert(tk.END, "2. 點擊 'LLaVA語義評分' 進行AI視覺+語義綜合評估\n")
        self.detail_text.insert(tk.END, "3. LLaVA會仔細觀察影像(70%)並結合AI描述(30%)評分\n")
        self.detail_text.insert(tk.END, "4. 以影像觀察為主，AI描述為輔助驗證\n")
        self.detail_text.insert(tk.END, "5. 選擇最佳候選位置後點擊導航按鈕")
        
        # 清空影像
        self.image_label.configure(image='', text="選擇候選位置以顯示影像")
    
    def update_candidates_display(self):
        """更新候選列表顯示"""
        # 清空候選列表
        for item in self.candidates_tree.get_children():
            self.candidates_tree.delete(item)
        
        # 添加候選結果
        for i, candidate in enumerate(self.current_candidates):
            rank = i + 1
            llava_score = f"{candidate['llava_score']}" if candidate['llava_score'] is not None else "未評分"
            pos = candidate['position']
            coords = f"({pos['x']:.1f}, {pos['y']:.1f}, {pos['z']:.1f})"
            ai_desc = candidate.get('ai_answer', '')[:60] + "..." if len(candidate.get('ai_answer', '')) > 60 else candidate.get('ai_answer', '')
            
            # 根據LLaVA分數設定行顏色
            tags = ()
            if candidate['llava_score'] is not None:
                if candidate['llava_score'] >= 80:
                    tags = ('high_score',)
                elif candidate['llava_score'] >= 60:
                    tags = ('medium_score',)
                elif candidate['llava_score'] < 40:
                    tags = ('low_score',)
            
            item_id = self.candidates_tree.insert('', 'end', values=(rank, llava_score, coords, ai_desc), tags=tags)
        
        # 設定標籤顏色
        self.candidates_tree.tag_configure('high_score', background='lightgreen')
        self.candidates_tree.tag_configure('medium_score', background='lightyellow')
        self.candidates_tree.tag_configure('low_score', background='lightcoral')
    
    def on_candidate_select(self, event):
        """候選位置選擇事件"""
        selection = self.candidates_tree.selection()
        if not selection:
            return
        
        item = self.candidates_tree.item(selection[0])
        rank = int(item['values'][0]) - 1
        
        if 0 <= rank < len(self.current_candidates):
            candidate = self.current_candidates[rank]
            self.selected_candidate = candidate  # 記錄選中的候選
            self.show_candidate_detail(candidate, rank + 1)
            
            # 更新導航狀態顯示
            if self.isaac_client:
                pos = candidate['position']
                self.nav_status_label.configure(
                    text=f"已選中位置: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}) - 點擊導航按鈕前往", 
                    foreground="blue"
                )
    
    def show_candidate_detail(self, candidate, rank):
        """顯示候選位置詳細資訊"""
        # 更新詳細資訊文本
        self.detail_text.delete(1.0, tk.END)
        
        detail_info = f"🎯 候選位置 #{rank}\n"
        detail_info += f"{'='*50}\n\n"
        
        # 分數資訊
        detail_info += f"📊 評分資訊:\n"
        if candidate['llava_score'] is not None:
            detail_info += f"   LLaVA語義評分: {candidate['llava_score']}/100\n"
            detail_info += f"   LLaVA置信度: {candidate['llava_confidence']}\n"
        else:
            detail_info += f"   LLaVA語義評分: 尚未評分\n"
        detail_info += f"\n"
        
        detail_info += f"🆔 ID: {candidate['id']}\n"
        detail_info += f"📅 時間戳: {candidate.get('timestamp', 'N/A')}\n\n"
        
        pos = candidate['position']
        detail_info += f"📍 位置座標:\n"
        detail_info += f"   X: {pos['x']:.6f}\n"
        detail_info += f"   Y: {pos['y']:.6f}\n"
        detail_info += f"   Z: {pos['z']:.6f}\n\n"
        
        rot = candidate['rotation']
        detail_info += f"🔄 旋轉 (四元數):\n"
        detail_info += f"   X: {rot['x']:.6f}\n"
        detail_info += f"   Y: {rot['y']:.6f}\n"
        detail_info += f"   Z: {rot['z']:.6f}\n"
        detail_info += f"   W: {rot['w']:.6f}\n\n"
        
        # LLaVA評分理由
        if candidate['llava_reasoning']:
            detail_info += f"🤖 LLaVA語義評估理由:\n{candidate['llava_reasoning']}\n\n"
        
        detail_info += f"❓ AI 問題:\n{candidate.get('ai_question', 'N/A')}\n\n"
        detail_info += f"💬 AI 回答:\n{candidate.get('ai_answer', 'N/A')}\n\n"
        
        if candidate.get('metadata'):
            detail_info += f"📋 元數據:\n{candidate['metadata']}\n\n"
        
        if candidate.get('camera_path'):
            detail_info += f"📷 影像路徑:\n{candidate['camera_path']}\n"
        
        self.detail_text.insert(tk.END, detail_info)
        
        # 顯示影像
        self.load_and_display_image(candidate.get('image_base64'))
    
    def load_and_display_image(self, base64_image):
        """載入並顯示影像"""
        if not base64_image:
            self.image_label.configure(image='', text="無影像資料")
            return
        
        try:
            image = self.searcher.decode_base64_image(base64_image)
            if image:
                image.thumbnail((280, 180), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo, text="")
                self.image_label.image = photo
            else:
                self.image_label.configure(image='', text="影像載入失敗")
                
        except Exception as e:
            self.image_label.configure(image='', text=f"影像載入錯誤: {str(e)}")
    
    def export_results(self):
        """匯出搜尋結果"""
        if not self.current_candidates:
            messagebox.showwarning("警告", "沒有可匯出的結果")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialname=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if filename:
            try:
                export_data = []
                for candidate in self.current_candidates:
                    export_candidate = candidate.copy()
                    if 'image_base64' in export_candidate:
                        del export_candidate['image_base64']
                    export_data.append(export_candidate)
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("成功", f"結果已匯出至: {filename}")
                
            except Exception as e:
                messagebox.showerror("錯誤", f"匯出失敗: {str(e)}")
    
    def run(self):
        """運行GUI應用程式"""
        try:
            logger.info("啟動 GUI 主循環...")
            self.root.mainloop()
        except Exception as e:
            logger.error(f"GUI 主循環錯誤: {e}")
            traceback.print_exc()

def main():
    """主程式 - 安全版本"""
    try:
        logger.info("開始創建應用程式...")
        
        # 設置安全的環境變數
        import os
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
        
        # 檢查基本環境
        logger.info("檢查Python環境...")
        logger.info("已移除 CLIP/PyTorch 依賴，使用純語義搜尋")
        logger.info("使用 ROS Bridge WebSocket 與 ros2_navigation extension 整合")
        
        # 創建應用程式
        app = SafeRobotVisionSearchGUI()
        
        logger.info("啟動應用程式...")
        app.run()
        
    except ImportError as e:
        logger.error(f"模組導入錯誤: {e}")
        print(f"錯誤：缺少必要的Python模組: {e}")
        print("請確保已安裝所有依賴項")
    except Exception as e:
        logger.error(f"主程式錯誤: {e}")
        traceback.print_exc()
        print(f"程式執行錯誤: {e}")
        print("請檢查錯誤日誌或聯繫技術支援")

if __name__ == "__main__":
    main()