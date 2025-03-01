# 等候時間預測
[![en](https://img.shields.io/badge/lang-en-red.svg)](https://github.com/I-Sheng/Traveling-Planner-With-LLM/blob/main/data/waiting_time/README.en.md)


## 內容概述

在規劃旅程時，考慮每個景點的停留時間是非常重要的。然而，透過 Google API 取得的資料並不總是包含等候時間。本專案的目標是利用熱門時段的資料，訓練一個模型來預測景點的等候時間，以便旅客能更妥善地安排行程。

考量到資料稀疏性的可能，我蒐集了嘉義市、嘉義縣及台南各景點的熱門時段資料，以建立和優化預測模型。

## 資料蒐集步驟

1. **安裝所需套件**

   - 安裝 [Populartimes 套件](https://github.com/m-wrzr/populartimes) 來透過 Google API 取得熱門時段及等候時間的資料。

2. **蒐集景點名稱以取得資料**

   - 使用以下觀光網站來獲取景點名稱：
     - [嘉義市觀光旅遊網](https://travel.chiayi.gov.tw/)
     - [嘉義縣政府文化觀光局](https://tbocc.cyhg.gov.tw/)
     - [台南旅遊網](https://www.twtainan.net/)

3. **取得熱門時段及等候時間資料**

   - 使用 Python 腳本來抓取並分類資料為四種類型：

     1. **nopopular_notimeSpent.json**：沒有熱門時段及等候時間的景點。
     2. **popular_notimeSpent.json**：只有熱門時段資料的景點。這部分資料是我們希望利用來預測等候時間的重點。
     3. **nopopular_timeSpent.json**：只有等候時間資料的景點。
     4. **popular_timeSpent.json**：包含熱門時段及等候時間的景點，可用於訓練及測試預測模型。

   - 抓取資料的指令範例：

     ```bash
     python3 getPopular_chiayi.py ./webcrab/chiayi_food.txt
     # 嘉義資料使用 getPopular_chiayi.py 腳本

     python3 getPopular_tainan.py ./webcrab/tainan_food.txt
     # 台南資料使用 getPopular_tainan.py 腳本
     ```

## 模型訓練

- 資料蒐集完成後，利用熱門時段資料訓練模型，預測景點的等候時間。
- 有關預測過程及模型效能的詳細說明，請參考 `prediction_model`。

## 未來改進

- 嘗試使用不同的機器學習模型及特徵工程，以改善等候時間預測的準確性。
- 擴展資料蒐集範圍，納入更多地區或景點，以提升模型的穩健性。
- 將模型整合到易於使用的工具或 API 中，方便旅客在規劃行程時查詢。
