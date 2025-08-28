# 機器人視覺記錄搜尋系統

一個基於語義搜尋和LLaVA視覺評估的智能機器人導航系統，支持通過自然語言指令搜尋相關位置並自動導航。

## 🌟 主要特點

### 🔍 純語義搜尋
- **關鍵字篩選**: 從AI描述中智能提取和匹配關鍵詞
- **移除CLIP依賴**: 不再依賴PyTorch和CLIP模型，提升系統輕量化
- **多語言支持**: 支持中文自然語言指令搜尋

### 🤖 LLaVA視覺評估
- **視覺+語義綜合評分**: 結合圖像觀察(70%)和AI描述驗證(30%)
- **智能評估**: 使用LLaVA模型對候選位置進行0-100分評分
- **置信度分析**: 提供high/medium/low置信度等級

### 🚀 ROS Bridge整合
- **WebSocket通訊**: 通過ROS Bridge與ros2_navigation extension整合
- **實時導航**: 支持即時發送導航指令到機器人
- **狀態監控**: 實時監控機器人導航狀態

### 💾 雙資料庫支持
- **Milvus**: 高性能向量資料庫支持
- **Qdrant**: 雲端向量資料庫支持
- **靈活切換**: 可在GUI中輕鬆切換資料庫類型

### 🎯 自動化功能
- **自動導航**: LLaVA評分後可自動導航到最佳位置
- **批量評估**: 支持批量評估多個候選位置
- **結果導出**: 支持將搜尋結果導出為JSON格式

## 📋 系統要求

### 基礎環境
- **Python**: 3.8+
- **操作系統**: Windows/Linux/macOS
- **記憶體**: 建議4GB以上

### 核心依賴
```
tkinter (GUI界面)
requests (HTTP通訊)
websocket-client (ROS Bridge通訊)
pymilvus (Milvus資料庫)
qdrant-client (Qdrant資料庫)
Pillow (圖像處理)
numpy (數值計算)
```

### 外部服務
- **LLaVA服務**: Ollama + LLaVA模型 (預設: http://localhost:11434)
- **ROS Bridge**: ROS Bridge服務器 (預設: ws://localhost:9090)
- **向量資料庫**: Milvus或Qdrant實例

## ⚡ 快速安裝

### 1. 安裝Python依賴
```bash
pip install tkinter requests websocket-client pymilvus qdrant-client Pillow numpy
```

### 2. 安裝LLaVA (使用Ollama)
```bash
# 安裝Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 下載LLaVA模型
ollama pull llava
```

### 3. 設置ROS Bridge
```bash
# 安裝rosbridge-suite
sudo apt-get install ros-<your-ros-distro>-rosbridge-suite

# 啟動ROS Bridge
roslaunch rosbridge_server rosbridge_websocket.launch
```

## 🔧 配置說明

### 資料庫配置

#### Milvus配置
```python
# 預設配置
Host: localhost
Port: 19530
Collection: ros2_camera_images
```

#### Qdrant配置
```python
# 預設配置
Host: localhost
Port: 6333
Collection: ros2_camera_images
API Key: (可選)
```

### LLaVA配置
```python
# Ollama LLaVA配置
Base URL: http://localhost:11434
Model: llava
```

### ROS Bridge配置
```python
# ROS Bridge WebSocket配置
URL: ws://localhost:9090
Robot Name: Baymax
Navigation Service: set_goal_pose
Status Topic: /baymax/navigation_status
```

### 導航指令格式

#### Service模式（實體機器人）
系統使用ROS Service調用進行導航，格式如下：

```json
{
  "op": "call_service",
  "service": "set_goal_pose",
  "args": {
    "task_mode": 0,
    "task_times": 1,
    "from_point": {
      "id": "",
      "x": 0.0,
      "y": 0.0,
      "theta": 0.0,
      "velocity_level": 1.0
    },
    "to_point": {
      "id": "target_1234567890",
      "x": 1.23,
      "y": 4.56,
      "theta": 0.0,
      "velocity_level": 1.0
    }
  },
  "id": "nav_request_1234567890"
}
```

**Service參數說明:**
- `task_mode`: 任務模式 (0=ONE_POINT單點導航)
- `task_times`: 執行次數
- `from_point`: 起始點座標 (通常為當前位置)
- `to_point`: 目標點座標
  - `id`: 點位標識符
  - `x`, `y`: 平面座標 (單位: 公尺)
  - `theta`: 朝向角度 (單位: 弧度)
  - `velocity_level`: 速度等級 (1.0=正常速度)

#### Topic模式 
系統也支援透過Topic發布導航目標，格式如下：

```json
{
  "op": "publish",
  "topic": "/baymax/navigation_goal",
  "msg": {
    "header": {
      "stamp": {
        "sec": 1234567890,
        "nanosec": 123456789
      },
      "frame_id": "world"
    },
    "pose": {
      "position": {
        "x": 1.23,
        "y": 4.56,
        "z": 0.0
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
```

**Topic訊息說明:**
- `header`: ROS標準訊息頭
  - `stamp`: 時間戳記
  - `frame_id`: 座標系統名稱 (通常為"world")
- `pose`: 目標姿態
  - `position`: 3D位置座標 (x, y, z)
  - `orientation`: 四元數朝向 (通常設為[0,0,0,1]表示無旋轉)

#### 狀態監控Topic
系統訂閱導航狀態更新：

```json
{
  "op": "subscribe",
  "topic": "/baymax/navigation_status",
  "type": "std_msgs/String"
}
```

狀態訊息內容包括：
- `"Idle"`: 待機狀態
- `"Navigating"`: 導航中
- `"Completed"`: 導航完成
- `"Failed"`: 導航失敗

## 🚀 使用說明

### 1. 啟動系統
```bash
python robot_vision_search.py
```

### 2. 連接服務
1. 在GUI中配置資料庫連接參數
2. 配置LLaVA API地址
3. 配置ROS Bridge地址
4. 點擊「重新連接」按鈕

### 3. 搜尋位置
1. 在指令輸入框中輸入自然語言指令，例如：
   - "請走到三角錐的位置"
   - "請走到桌子那"
   - "走到架子旁邊"

2. 點擊「🔍 獲取候選」按鈕進行關鍵字搜尋

3. 點擊「🤖 LLaVA視覺+語義評分」進行智能評估

### 4. 選擇和導航
1. 在候選位置列表中選擇目標位置
2. 查看詳細信息和評分理由
3. 點擊「🚀 導航到選中位置」執行導航

### 5. 自動化功能
- 勾選「LLaVA評分後自動導航」實現全自動搜尋導航
- 使用常用指令快捷按鈕快速發起搜尋

## 🏗️ 系統架構

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GUI Interface │    │  Search Engine  │    │ Database Layer  │
│                 │◄──►│                 │◄──►│                 │
│ • 指令輸入       │    │ • 關鍵字提取      │    │ • Milvus       │
│ • 結果顯示       │    │ • 候選搜尋        │    │ • Qdrant       │
│ • 參數設定       │    │ • 結果排序        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ LLaVA Evaluator │    │  ROS Bridge     │    │   Navigation    │
│                 │    │                 │    │                 │
│ • 視覺分析       │    │ • WebSocket     │    │ • 路徑規劃       │
│ • 語義理解       │    │ • Service Call  │    │ • 狀態監控       │
│ • 置信度評估     │     │ • 狀態訂閱       │    │ • 自動導航       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 核心組件說明

#### BaseVisionSearcher (搜尋引擎基類)
- 抽象介面定義
- 支援多種資料庫實現
- 統一的搜尋API

#### RobotVisionSearcher (Milvus實現)
- Milvus向量資料庫操作
- 關鍵字提取和匹配
- 候選位置篩選

#### QdrantVisionSearcher (Qdrant實現)
- Qdrant向量資料庫操作
- 與Milvus相同的搜尋邏輯
- 雲端部署支持

#### OllamaLLaVAClient (視覺評估器)
- LLaVA模型整合
- 圖像+文本綜合分析
- 智能評分算法

#### IsaacSimClient (ROS Bridge客戶端)
- WebSocket通訊
- ROS Service調用
- 導航狀態監控

## 🔧 故障排除

### 常見問題

#### 1. 資料庫連接失敗
```
解決方案：
- 檢查資料庫服務是否啟動
- 確認連接參數正確
- 檢查網絡連接
- 驗證Collection是否存在
```

#### 2. LLaVA評估失敗
```
解決方案：
- 確認Ollama服務運行正常
- 檢查LLaVA模型是否下載完成
- 驗證API地址配置
- 檢查圖像數據完整性
```

#### 3. ROS Bridge連接問題
```
解決方案：
- 檢查ROS Bridge服務狀態
- 確認WebSocket地址正確
- 驗證ROS環境變量
- 檢查防火牆設置
```

#### 4. 導航指令無響應
```
解決方案：
- 確認ros2_navigation extension運行
- 檢查Service名稱配置
- 驗證機器人狀態
- 查看ROS日誌錯誤
```

### 調試技巧

#### 啟用詳細日誌
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 檢查服務狀態
```bash
# 檢查ROS Bridge
rostopic list
rosservice list

# 檢查資料庫連接
telnet localhost 19530  # Milvus
telnet localhost 6333   # Qdrant
```

## 📊 性能優化

### 搜尋優化
- 調整`top_k`參數控制候選數量
- 使用關鍵字預篩選減少LLaVA評估量
- 開啟結果緩存提升重複搜尋速度

### LLaVA評估優化
- 降低`temperature`參數提高評分一致性
- 批量處理減少API調用開銷
- 異步評估提升用戶體驗

### 導航優化
- 移除導航間隔限制（已在代碼中實現）
- 使用狀態監控避免重複指令
- 實現路點緩衝優化導航路徑
