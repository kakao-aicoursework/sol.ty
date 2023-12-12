#-*- coding: utf-8 -*-


# SimpleText : https://chatbot.kakao.com/docs/skill-response-format#simpletext
simple_text_sample = {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "simpleText": {
                    "text": "안녕하세요! 저는 챗봇입니다."
                }
            },
            {
                "simpleText": {
                    "text": "어떤 카드를 보여드릴까요?"
                }
            },
        ],
        "quickReplies": [
            {
                "messageText": "Basic Card 보여주세요",
                "action": "message",
                "label": "Basic"
            },
            {
                "messageText": "Commerce Card 보여주세요",
                "action": "message",
                "label": "Commerce"
            },
        ]
    }
}

# BasicCard : https://chatbot.kakao.com/docs/skill-response-format#basiccard
basic_card_sample = {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "basicCard": {
                    "title": "라이언",
                    "description": "덩치는 크지만 마음은 여린 수사자",
                    "thumbnail": {
                        "imageUrl": "https://t1.kakaocdn.net/friends/new_store/prod/character/character_20230609082239_4d31bb9f1570488fa272c6c3f62ead6c.jpg"
                    },
                    "buttons": [
                        {
                            "label": "더 알아보기",
                            "action": "webLink",
                            "webLinkUrl": "https://namu.wiki/w/%EB%9D%BC%EC%9D%B4%EC%96%B8(%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%94%84%EB%A0%8C%EC%A6%88)"
                        }
                    ]
                }
            },
        ]
    }
}

# Commerce Card : https://chatbot.kakao.com/docs/skill-response-format#commercecard
commerce_card_sample = {
    "version": "2.0",
    "template": {
        "outputs": [
            {
                "commerceCard": {
                    "title": "",
                    "description": "두 뺨이 발그레😊 매일쓰는 칫솔을 깨끗하게!",
                    "price": 25000,
                    "discountRate": 20,
                    "discountedPrice": 20000,
                    "currency": "won",
                    "thumbnails": [
                        {
                            "imageUrl": "https://t1.kakaocdn.net/friends/prod/product/20230620141231526_8809922502300_AW_00.jpg",
                            "link": {
                                "web": "https://store.kakaofriends.com/products/9959"
                            }
                        }
                    ],
                    "buttons": [
                        {
                            "label": "구매하기",
                            "action": "webLink",
                            "webLinkUrl": "https://store.kakaofriends.com/products/9959"
                        }
                    ]
                }
            },
        ]
    }
}


list_card = {
  "version": "2.0",
  "template": {
    "outputs": [
      {
        "listCard": {
          "header": {
            "title": "챗봇 관리자센터를 소개합니다."
          },
          "items": [

            {
              "title": "카카오T",
              "description": "모든 이동을 위한, 카카오 T",
              "imageUrl": "https://t1.kakaocdn.net/kakaocorp/kakaocorp/admin/4086aee3017800001.png",
              "link": {
                "web": "https://www.kakaocorp.com/page/service/service/KakaoT"
              }
            },
                        {
              "title": "카카오Pay",
              "description": "금융의 수고로움을 줄이고 생활에 이로운 흐름을 만듭니다.",
              "imageUrl": "https://t1.kakaocdn.net/kakaocorp/kakaocorp/admin/5e5fc297017800001.png",
              "link": {
                "web": "https://www.kakaocorp.com/page/service/service/KakaoPay"
              }
            },
                        {
              "title": "카카오뱅크",
              "description": "같지만 다른 은행, 카카오뱅크",
              "imageUrl": "https://t1.kakaocdn.net/kakaocorp/kakaocorp/admin/40bd741c017800001.png",
              "link": {
                "web": "https://www.kakaocorp.com/page/service/service/KakaoBank"
              }
            },

          ],
          "buttons": [
            {
              "label": "서비스 더 보기",
              "action": "webLink",
              "webLinkUrl": "https://www.kakaocorp.com/page/service/service"
            }
          ]
        }
      }
    ]
  }
}