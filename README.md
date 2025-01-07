## Introduction
This project introduces a novel travel itinerary planning system leveraging Large Language Models (LLMs) like GPT-4. The system addresses challenges in self-planned travel by simplifying complex tasks such as route optimization, time management, and preference alignment. Key features include personalized recommendations, predictive stay time modeling, and itinerary optimization using the Vehicle Routing Problem with Time Windows (VRPTW) algorithm.

## System Architecture
![image](https://github.com/user-attachments/assets/95264cd2-dd9d-4b99-a1df-2d2a3fa5f965)

## Contributions
1. Utilized LLMs with RAG to recommend themed Points of Interest (POIs) tailored to user preferences
2. Developed predictive models for estimating stay times at POIs, ensuring itineraries respect critical time constraints
3. Implemented the VRPTW algorithm for optimized travel itinerary planning
4. Developed an advanced online platform for personalized and time-efficient travel itinerary planning.

## System Website
### Desktop version
![computer 1](https://github.com/user-attachments/assets/84557442-5843-4ae4-9bbe-9b4d20631591)
![computer 2](https://github.com/user-attachments/assets/cbcfc321-d6ab-4932-aaeb-aa4a9eaed621)
![computer 3](https://github.com/user-attachments/assets/ee30290a-287c-4771-9398-1087ab56d5df)
![computer 4](https://github.com/user-attachments/assets/9015a93d-fd9f-429f-bc0d-d81c0f28947f)
![computer 5](https://github.com/user-attachments/assets/61e2b203-7189-42c5-b89c-2559820ba0d9)
![computer 6](https://github.com/user-attachments/assets/ad22941f-7907-40a8-b5ce-7b21499258ca)
### Mobile version

## 環境設定

1. **建立 `.env` 檔案**
   在專案根目錄下建立 `.env` 檔案，內容如下：
    ```env
    # 生成式 AI
    OPENAI_API_KEY='YOUR_OPENAI_API_KEY'
    ANTHROPIC_API_KEY='YOUR_ANTHROPIC_API_KEY'
    GOOGLE_API_KEY='YOUR_GOOGLE_API_KEY'

    # Google 地圖 API
    GOOGLE_MAP_API_KEY='YOUR_GOOGLE_MAP_API_KEY'

    # LangChain
    LANGCHAIN_HUB_API_KEY='YOUR_LANGCHAIN_HUB_API_KEY'
    LANGCHAIN_API_KEY='YOUR_LANGCHAIN_API_KEY'
    ```

