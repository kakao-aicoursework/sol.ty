# Kakao Chatbot Skill Example (생성 AI의 이해와 LLM 적용 및 실습 - 2)
[카카오 챗봇빌더](https://chatbot.kakao.com/)에 연결할 스킬 서버를 쉽게 개발할 수 있도록 참고할 수 있는 예제입니다. by mario.h

## Skill
아래는 간단한 스킬서버 예제입니다. 카카오 챗봇빌더 서버의 [HTTP POST 요청](https://chatbot.kakao.com/docs/skill-response-format#skillpayload)을 받아서, 원하는 [응답](https://chatbot.kakao.com/docs/skill-response-format#skillresponse)을 만들 수 있습니다.


```python

app = FastAPI()

@app.post("/skill/hello")
async def skill(req: ChatbotRequest):

    response_json = {

        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": "안녕하세요!"
                    }
                }
            ]
        }
    }

    return response_json

```





## 콜백
일반적인 스킬은 타임아웃이 5초로 설정되어 있습니다.  
5초 이상의 처리 시간이 필요한 경우에는 콜백을 이용하여 구현할 수 있습니다.   
아래는 콜백이 동작하는 방식입니다.   
스킬 요청을 받아서 (1) 콜백 활성화 (2) 응답 메시지 전송, 이렇게 두 단계로 사용자에게 응답을 보낼 수 있습니다.

```
     ┌─────┐          ┌────────────┐     
     │Kakao│          │Skill Server│     
     └──┬──┘          └─────┬──────┘     
        │     Request       │            
        │──────────────────>│            
        │                   │            
        │ Enable Callback   │            
        │<──────────────────│            
        │                   │            
        │         ╔═════════╧═══════════╗
        │         ║Time consuming task ░║
        │         ╚═════════╤═══════════╝
        │     Callback      │            
        │<──────────────────│   
```


1. 콜백 활성화
    ```json
    {
            "version" : "2.0",
            "useCallback" : true
    }
    ```

2. 응답 메시지 전송
(1)에서 요청에 포함된 ```userRequest.callbackUrl```로 HTTP POST 요청을 보냅니다.


# 스킬서버 설정하기
ifconfig를 통해 확인한 개별 ip를 서버 주소로 설정합니다.    
교육 종료 후 계속 사용을 원하신다면 해당 코드를 private한 타 서버에 배포 후 스킬 연결 부탁드립니다.  
(현재 개인 ip를 통한 스킬 서버 등록은 사내망에서만 가능합니다.)




## 스킬 만들기
예제에는 4개의 endpoint가 만들어져 있습니다.  
이것은 각각 다음과 같이 Full URL로 바꿀 수 있습니다.
```
/skill/hello  -->  https://{ip:port}/skill/hello
/skill/basic-card --> https://{ip:port}/skill/basic-card
/skill/commerce-card --> https://{ip:port}/skill/commerce-card
/callback --> https://{ip:port}/callback <<<<<<<<< 활용!

```

## 블록에 스킬 서버 연결하기 
1. 위에서 만든 ```My Skill```을 선택합니다.
2. 봇응답을 ```스킬데이터 사용```으로 변경합니다.
3. 저장

<img width="759" alt="image" src="https://github.com/mariojisoohwang/kakao-chatbot-skill-example/assets/970595/9c8fbdab-aa01-400f-bcf5-35d44723f79c">


## 콜백
callback.py 참고

