import os
import time
import sys
import glob
import pandas as pd
from sqlalchemy import create_engine, text
import torch
from sentence_transformers import SentenceTransformer

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/spotsync"
MODEL_NAME = "jhgan/ko-sroberta-multitask"
DEVICE = "cpu"

# Category-specific semantic descriptions and keywords to solve semantic dilution
CATEGORY_DESCRIPTIONS = {
    # 1. Cafes, Dessert & Study
    "카페": "신선한 원두로 내린 풍미 가득한 아메리카노와 달콤한 카페라떼, 디저트를 맛보며 여유롭게 대화하거나 편안하게 머무르기 좋은 아늑한 감성의 카페 커피숍 공간입니다.",
    "독서실/스터디 카페": "조용하고 차분한 면학 분위기 속에서 학습에 집중할 수 있도록 개별 칸막이 독서실 책상, 백색소음기, 스터디룸 및 편리한 콘센트를 완비한 최적의 스터디 카페 공부방입니다.",
    "빵/도넛": "매일 아침 매장에서 갓 구워낸 바삭하고 고소한 소금빵, 크루아상, 케이크, 도넛 등 다채로운 프리미엄 베이커리와 디저트를 맛볼 수 있는 맛있는 빵집 제과점 공간입니다.",
    "토스트/샌드위치/샐러드": "신선하고 아삭한 야채와 햄, 치즈로 가득 채운 영양 만점의 수제 샌드위치, 갓 구운 토스트, 그리고 다이어트에 좋은 신선한 샐러드를 간편하게 즐기는 웰빙 브런치 맛집입니다.",
    "아이스크림/빙수": "더위를 시원하게 식혀주는 달콤하고 쫀득한 젤라또 아이스크림과 팥, 과일, 인절미가 풍성하게 올라간 시원하고 부드러운 우유 눈꽃 빙수 전문점 디저트 카페입니다.",
    "서점": "문학, 경제, 인문, 수험서 등 다채로운 신간 도서가 가득하며, 조용하게 책을 읽고 사색하거나 트렌디한 잡지와 학용품을 구경하고 문화적 감성을 채우기 좋은 힐링 서점 공간입니다.",

    # 2. PC rooms & Gaming
    "PC방": "최상급 그래픽카드(RTX 시리즈)와 초고속 인터넷, 고주사율 게이밍 모니터, 인체공학적인 넓은 게이밍 의자를 완비하여 끊김 없이 몰입해서 컴퓨터 게임을 즐기기 좋은 프리미엄 PC방 공간입니다.",
    "전자 게임장": "다양한 체감형 아케이드 게임, 리듬 게임, 레이싱 및 펌프 게임기 등이 가득하여 친구들과 스트레스를 날려버리고 즐거운 추억을 쌓을 수 있는 활기찬 전자 오락실 게임장입니다.",
    "바둑/장기/체스 경기 운영업": "정숙하고 진지한 분위기 속에서 차분하게 바둑, 장기, 체스 대국을 두며 두뇌 싸움의 묘미를 즐기고 친목을 도모하기 좋은 전통 놀이 경기 기원 대국실 공간입니다.",

    # 3. Music & Vocal rehearsal
    "노래방": "풍부한 음질의 최신 반주기와 무선 마이크, 화려한 LED 미러볼 조명을 갖춰 신나게 노래를 부르고 스트레스를 완전히 해소할 수 있는 즐거운 코인 노래연습장 노래방 공간입니다.",
    "음악학원": "피아노, 보컬, 기타, 드럼 등 체계적인 1대1 맞춤 강습을 제공하며, 초보자부터 전문가까지 기초를 탄탄히 다지고 실력을 향상할 수 있는 전문 음악 지도 교육 아카데미입니다.",
    "합주실": "최고 사양의 드럼 세트, 앰프, 마이크와 수준 높은 방음 부스가 완비되어 밴드 동호회가 완벽한 하모니로 악기 연주와 단체 합주 연습을 진행할 수 있는 전문 밴드 음악 합주 공간입니다.",
    "연습실": "보컬 연습, 악기 연주, 댄스 안무 연습 등을 주변 소음 걱정 없이 프라이빗하고 조용하게 연습할 수 있도록 쾌적하게 설계된 1인 및 단체용 방음 대여 연습실 공간입니다.",

    # 4. Fast food & Hamburgers
    "버거": "두툼하고 육즙 가득한 수제 쇠고기 패티와 신선한 야채, 특제 소스가 어우러진 맛있는 프리미엄 햄버거와 바삭한 감자튀김을 맛볼 수 있는 버거 전문 맛집 식당입니다.",
    "피자": "쫄깃하고 담백한 도우 위에 풍성한 치즈와 페퍼로니, 고구마 등 다채로운 토핑을 올려 갓 구워낸 맛있는 정통 피자 전문 식당입니다.",
    "치킨": "바삭하고 고소하게 튀겨낸 후라이드 치킨과 매콤달콤한 특제 양념 치킨, 달콤 짭조름한 간장 치킨 등을 즐길 수 있는 대한민국 대표 치킨 맥주 치맥 맛기 식당입니다.",
    "그 외 기타 간이 음식점": "핫도그, 떡꼬치, 닭강정 등 부담 없이 빠르게 맛볼 수 있는 다채로운 스트리트 푸드와 간편식을 판매하는 아기자기한 간이 음식점 식당입니다.",

    # 5. Korean meals & Hot Soups
    "백반/한정식": "계절별로 정성스럽게 차려낸 정갈한 나물 반찬과 찌개, 갓 지은 따뜻한 밥이 어우러져 어머니의 집밥처럼 든든하고 건강한 한 끼를 대접받는 전통 한정식 백반 맛집 밥집입니다.",
    "국/탕/찌개류": "깊고 진하게 우려낸 육수에 고기와 야채를 듬뿍 넣어 끓여낸 뜨끈하고 얼큰한 김치찌개, 된장찌개, 해장국, 갈비탕 등으로 지친 몸을 든든하게 채우는 전통 국탕 찌개 한식 식당입니다.",
    "국수/칼국수": "진한 멸치 육수나 사골 육수에 쫄깃한 생면을 말아낸 따뜻한 칼국수와 새콤달콤하게 비벼낸 잔치국수, 비빔국수를 부담 없이 맛있게 호로록 즐길 수 있는 전통 면요리 국수 맛집입니다.",
    "김밥/분식": "고소하게 말아낸 통통한 수제 김밥과 매콤달콤한 국민 간식 떡볶이, 바삭한 모듬 튀김, 찰순대를 한자리에서 푸짐하게 맛볼 수 있는 친근하고 맛있는 분식집 맛집 공간입니다.",
    "분식": "매콤 쫄깃한 떡볶이, 바삭하고 고소한 김말이 튀김, 따끈한 어묵 국물과 순대 등 남녀노소 누구나 좋아하는 국민 간식을 맛있게 즐길 수 있는 전통 분식 맛집 식당입니다.",
    "김밥/만두/분식": "속이 꽉 찬 수제 왕만두와 담백한 찐만두, 알차게 말아낸 가성성 좋은 김밥 및 라면, 돈가스를 다양하고 든든하게 주문해 먹기 좋은 만두 분식 맛집 전문점입니다.",
    "닭/오리고기 구이/찜": "매콤달콤하게 볶아낸 철판 닭갈비, 한방 약재를 넣어 고아낸 닭백숙, 그리고 고소하게 구워낸 오리구이로 환절기 기력을 든든하게 보충하는 건강 보양 닭오리 전문 식당입니다.",
    "돼지고기 구이/찜": "지글지글 노릇하게 구워낸 삼겹살, 목살, 갈비와 푹 끓여내 부드러운 매운 돼지갈비찜을 정갈한 반찬과 함께 쌈 싸 먹으며 모임하기 가장 좋은 돼지고기 구이 전문 식당입니다.",
    "소고기 구이/찜": "마블링이 훌륭하여 입안에서 부드럽게 사르르 녹는 프리미엄 소갈비, 등심, 살치살 구이와 달콤하고 짭조름한 소불고기 전골을 즐기는 고급 한식 소고기 전문 레스토랑 식당입니다.",
    "곱창 전골/구이": "고소한 곱이 꽉 찬 소곱창, 대창, 막창 구이와 칼칼하고 진한 국물 맛이 일품인 곱창전골에 술 한잔을 곁들이며 하루의 피로를 풀기 좋은 곱창 전문 한식 맛집입니다.",
    "족발/보쌈": "한약재와 함께 푹 삶아내어 야들야들하고 쫄깃한 콜라겐 가득 족발과 기름기를 쏙 빼 담백하고 부드러운 수육 보쌈을 매콤한 보쌈김치와 곁들여 먹는 족발보쌈 맛집입니다.",
    "해산물 구이/찜": "키조개, 가리비 등 신선한 조개구이와 아삭한 콩나물을 듬뿍 넣어 매콤하고 푸짐하게 쪄낸 아구찜, 해물찜을 바다 내음 가득 즐길 수 있는 신선한 해산물 전문 요리 식당입니다.",
    "횟집": "수조에서 갓 꺼내어 활어로 썬 쫄깃하고 신선한 광어, 우럭 회와 전복, 멍게 등 푸짐한 스끼다시, 마지막으로 얼큰하고 개운한 매운탕까지 풀코스로 맛보는 정통 활어 횟집입니다.",
    "냉면/밀면": "살얼음 동동 띄운 시원하고 깔끔한 육수의 물냉면과 매콤새콤한 비빔양념장이 중독성 있는 비빔냉면, 부산 전통의 쫄깃한 밀면으로 더위를 한방에 날려주는 냉면 전문 식당입니다.",
    "전/부침개": "노릇노릇하고 바삭하게 부쳐낸 해물파전, 김치전, 감자전에 달달하고 시원한 막걸리 한잔을 곁들이며 비 오는 날 운치 있게 대화를 나누고 모임하기 좋은 전통 민속 전집 주점입니다.",
    "기타 한식 음식점": "정형화되지 않은 이색적인 퓨전 한식과 지역 토속 음식을 정갈하게 재해석하여 남녀노소 모두의 입맛을 사로잡는 숨겨진 동네 한식 맛집 식당 공간입니다.",
    "떡/한과": "명절이나 특별한 날에 어울리는 쫄깃한 송편, 인절미, 시루떡과 달콤한 약과, 한과 등 정성껏 빚어낸 한국 전통 웰빙 떡 디저트 전문 떡집 공간입니다.",
    "정육점": "산지 직송한 신선하고 등급 높은 한우와 한돈을 부위별로 철저하게 관리하여 믿고 구매할 수 있는 정직한 동네 정육점 고기 판매점입니다.",

    # 6. Foreign restaurants & Fine dining
    "중국집": "달콤하고 춘장 풍미 가득한 짜장면과 해물이 듬뿍 들어가 얼큰하고 칼칼한 짬뽕, 겉바속촉의 대명사 탕수육을 맛있게 요리해 제공하는 친근한 중식당 중국요리 맛집입니다.",
    "파스타/스테이크": "은은한 조명과 세련된 인테리어 속에서 정통 이탈리안 크림/토마토 파스타와 부드럽고 육즙 가득한 안심 스테이크, 와인을 곁들여 소개팅이나 데이트를 즐기는 이탈리안 레스토랑 맛집입니다.",
    "일식 면 요리": "오랜 시간 푹 우려낸 돈사골 육수에 쫄깃한 차슈가 올라간 일본식 라멘, 쯔유 소스에 찍어 먹는 시원한 소바, 탱글탱글한 우동을 맛보는 정통 일식 면 전문점 맛집입니다.",
    "일식 카레/돈가스/덮밥": "겉은 바삭하고 속은 부드러운 육즙의 겉바속촉 수제 돈카츠, 진하고 깊은 일본식 고체 카레, 짭조름하고 맛있는 규동 및 가츠동 덮밥을 즐기는 깔끔하고 아기자기한 일식당 맛집입니다.",
    "일식 회/초밥": "신선하게 쥔 정통 초밥(스시)과 두툼하고 부드러운 모듬 사시미 회를 정갈하게 대접받으며 소규모 모임이나 격식 있는 식사를 진행하기 좋은 깔끔하고 세련된 일식 초밥 맛집입니다.",
    "기타 서양식 음식점": "그리스, 스페인 등 개성 넘치는 유럽식 타파스 요리나 멕시칸 타코, 퀘사디아 등 이색적이고 트렌디한 서양 요리를 감성적인 분위기 속에서 즐길 수 있는 양식 레스토랑 맛집입니다.",
    "기타 일식 음식점": "꼬치구이 전문 야키토리나 일본 정통 요리를 캐주얼하고 분위기 있게 풀어내어 하이볼 한잔을 가볍게 곁들여 대화를 나누기 좋은 이색 일식 맛집 공간입니다.",
    "베트남식 전문": "장시간 우려낸 깊고 시원한 육수의 소고기 쌀국수(포)와 새콤달콤한 소스에 찍어 먹는 분짜, 바삭한 스프링롤 등 이국적이고 매력적인 베트남 요리를 전문으로 하는 아시안 맛집입니다.",
    "기타 동남아식 전문": "태국의 팟타이, 똠얌꿍, 푸팟퐁커리 및 인도네시아 나시고랭 등 특색 있는 향신료와 재료들이 어우러진 동남아시아 전통 음식을 맛볼 수 있는 이국적인 맛집 식당입니다.",
    "분류 안된 외국식 음식점": "터키 케밥, 인도 정통 커리와 난 등 전 세계의 다채롭고 독특한 전통 요리들을 현지의 맛 그대로 재현해 내어 특별한 미식 경험을 선물하는 이색 외국식 식당입니다.",
    "뷔페": "한식, 중식, 일식, 양식부터 달콤한 디저트 베이커리까지 엄선된 수십 가지의 요리를 한자리에서 취향껏 무제한으로 마음껏 즐길 수 있는 고급 패밀리 뷔페 레스토랑 공간입니다.",

    # 7. Bars & Nightlife
    "요리 주점": "트렌디한 음악과 감성적인 어두운 조명 아래, 단순한 마른안주가 아닌 셰프의 고품격 요리 안주와 시원한 하이볼, 전통주, 소주 한잔을 즐기며 대화하기 좋은 분위기 좋은 요리 주점입니다.",
    "생맥주 전문": "갓 뽑아내어 머리끝까지 시원하고 청량한 프리미엄 크림 생맥주와 바삭하고 짭조름한 치킨, 모듬 소시지 안주를 부담 없이 즐기며 스트레스를 푸는 캐주얼 맥주 호프 전문점입니다.",
    "일반 유흥 주점": "화려한 사이키 조명과 신나는 사운드 속에서 다양한 주류와 안주를 주문해 지인들과 흥겨운 밤을 보내고 일상의 스트레스를 마음껏 발산할 수 있는 유흥 펍 주점 공간입니다.",
    "무도 유흥 주점": "쿵쾅거리는 웅장한 비트의 클럽 음악에 맞춰 신나게 춤을 추고 화려한 무대 조명 속에서 친구들과 흥겹게 뛰어놀며 밤을 불태울 수 있는 에너지 넘치는 댄스 클럽 주점입니다.",
    "주류 소매업": "전 세계의 엄선된 내추럴 와인, 빈티지 위스키, 수제 맥주와 개성 있는 전통주를 종류별로 갖추고 있으며 홈파티나 선물용 술을 전문적으로 추천받아 구매할 수 있는 와인바 보틀숍입니다.",

    # 8. Sports & Fitness
    "헬스장": "최신 웨이트 머신과 유산소 러닝머신, 천국의 계단 기구를 갖추고 전문 트레이너의 1대1 PT(피티)를 통해 근력 향상과 체력 증진, 건강한 다이어트를 실천하는 활기찬 헬스 피트니스 센터입니다.",
    "요가/필라테스 학원": "차분한 음악 속에서 몸의 균형과 속근육을 강화하고 자세 교정, 유연성 향상, 다이어트 및 심신 안정을 돕는 전문 기구 필라테스 및 요가 클래스 운동 스튜디오입니다.",
    "체형/비만 관리": "체계적인 인바디 측정과 맞춤형 식이 지도, 슬리밍 관리 기기를 통해 건강하고 체계적으로 비만 관리를 받고 아름다운 바디라인을 가꿀 수 있는 전문 체형 뷰티 케어 샵입니다.",
    "종합 스포츠시설": "수영, 농구, 실내 풋살, 배드민턴 등 다양한 구기 종목과 레저 스포츠를 날씨 제한 없이 한곳에서 쾌적하게 즐길 수 있는 대규모 종합 체육관 스포츠 시설입니다.",
    "스포츠/레크리에이션 용품 대여업": "캠핑 용품, 등산 기구, 자전거, 인라인스케이트 등 야외 스포츠와 레저 활동에 필요한 장비들을 합리적인 가격에 전문적으로 대여해 주는 스포츠 장비 렌탈 전문 샵입니다.",
    "당구장": "깔끔하게 관리된 3쿠션 대대와 중대 당구대, 포켓볼 다이를 완비하여 친구들이나 동료들과 함께 내기 당구를 치며 건전한 친목 도모와 여가를 즐기기 좋은 쾌적한 당구클럽입니다.",
    "탁구장": "가벼운 탁구채와 통통 튀는 탁구공으로 단수 및 복식 경기를 치르며 땀 흘려 운동하고, 순발력과 체력을 유쾌하게 기를 수 있는 실내 탁구장입니다.",
    "볼링장": "화려한 락 조명과 신나는 음악이 흐르는 레인 위로 묵직한 볼링공을 굴려 스트라이크를 치며 짜릿한 쾌감을 맛보고 가벼운 음주도 곁들일 수 있는 락 볼링장 스포츠 공간입니다.",
    "수영장": "맑고 깨끗하게 수질 관리된 다인용 레인과 유아풀을 갖추어, 무더운 날 시원하게 물놀이를 즐기거나 전신 유산소 운동인 수영을 체계적으로 배울 수 있는 실내 공공 수영장 시설입니다.",
    "골프 연습장": "초정밀 스크린 골프 센서 타석과 실외 인도어 타석을 갖추어, 초보 골퍼의 스윙 자세 교정부터 실전 필드 라운딩 연습까지 쾌적하게 골프를 즐기는 스크린 골프존 연습장입니다.",
    "테니스장": "사계절 내내 날씨 걱정 없이 랠리를 주고받을 수 있는 실내 코트와 코트장 등을 완비하여 활기차게 유산소 코트 플레이를 즐기는 전문 테니스 클럽 코트입니다.",
    "스쿼시/라켓볼장": "벽면에 라켓 공을 강력하게 튕기며 엄청난 활동량과 칼로리 소모를 자랑하는 실내 스포츠 스쿼시를 역동적이고 유쾌하게 배우고 땀 흘려 즐길 수 있는 전문 스쿼시 코트 센터입니다.",
    "태권도/무술학원": "대한민국 전통 무술인 태권도의 품새와 발차기, 격파는 물론, 유도나 합기도 등 호신술과 신체 단련, 예절 교육을 지도하는 어린이 및 성인 대상 전문 무도 무술 체육관입니다.",
    "기타 스포츠시설 운영업": "실내 암벽등반(클라이밍), 양궁 연습장, 펜싱 등 평소 접하기 힘든 독특하고 이색적인 실내 스포츠를 안전하고 체계적으로 배우며 취미 활동을 즐길 수 있는 이색 스포츠 시설입니다.",

    # 9. Beauty, Relaxation & Salons
    "미용실": "실력 있는 전문 헤어 디자이너가 고객의 두상과 모발에 맞게 트렌디한 헤어 컷, 볼륨 펌, 애쉬 염색, 클리닉을 1대1 맞춤 스타일링하여 헤어스타일을 예쁘게 가꾸는 전문 헤어살롱 미용실입니다.",
    "네일숍": "섬세한 터치로 손톱 기본 케어부터 화려한 젤 네일 아트, 패디큐어, 각질 관리를 통해 손끝과 발끝을 아름답고 위생적으로 꾸미며 힐링할 수 있는 프리미엄 전문 네일샵 뷰티 숍입니다.",
    "피부 관리실": "지친 피부에 수분과 영양을 공급하는 전문 에스테틱 페이셜 스킨케어, 여드름 진정 케어 및 경락 경혈 관리를 통해 맑고 탄력 있는 꿀피부를 가꾸는 전문 뷰티 피부 관리 에스테틱 샵입니다.",
    "마사지/안마": "오랜 시간 쌓인 만성 피로와 뭉친 근육을 부드러운 아로마 테라피나 타이 마사지, 지압 안마를 통해 말끔하게 풀어주고 깊은 신체 릴랙싱과 쉼을 선물하는 전문 테라피 안마 관리실입니다.",
    "목욕탕/사우나": "뜨끈한 온탕과 열탕, 땀을 흠뻑 흘릴 수 있는 황토 불가마 찜질방 시설을 완비하여, 온몸의 피로를 시원하게 풀고 상쾌하게 세신 목욕을 즐기는 도심 속 힐링 사우나 목욕탕 시설입니다."
}

def get_semantic_desc(name, major, mid, sub):
    full_name = name.strip()
    
    desc = CATEGORY_DESCRIPTIONS.get(sub)
    if not desc:
        desc = f"전문성과 오랜 경험을 바탕으로 친절하고 체계적인 일대일 서비스를 제공하여 방문하시는 모든 고객에게 깊은 만족을 선사하는 신뢰도 높고 깔끔한 {major} {mid} {sub} 관련 비즈니스 전문 공간입니다."
        
    return f"{full_name}은 {sub} ({major} > {mid}) 업종에 속하며, {desc}"

def main():
    print("="*80)
    print("      SpotSync AI - Advanced V3 Semantic Context Embedding Generator")
    print("="*80)
    
    engine = create_engine(DATABASE_URL)
    
    # 1. Add V3 columns
    print("[DB] Ensuring 'embedding_text_v3' & 'embedding_vector_v3' columns exist...")
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_text_v3 Text;"))
        conn.execute(text("ALTER TABLE places ADD COLUMN IF NOT EXISTS embedding_vector_v3 float8[];"))
        
    # 2. Get active V1 places
    print("[DB] Fetching rows with active V1 embeddings...")
    with engine.connect() as conn:
        existing_rows = conn.execute(text(
            "SELECT id, place_id, name, category, address FROM places WHERE embedding_vector IS NOT NULL"
        )).all()
        
    print(f"[DB] Found {len(existing_rows)} active rows.")
    if not existing_rows:
        print("[ERROR] No active rows to migrate.")
        return
        
    existing_place_ids = {row[1] for row in existing_rows}
    
    # 3. Locate CSV
    print("[CSV] Locating Gyeonggi-do CSV file...")
    csv_files = glob.glob("*.csv")
    gyeonggi_files = [f for f in csv_files if "경기" in f]
    csv_file_path = gyeonggi_files[0] if gyeonggi_files else csv_files[0]
    
    print(f"[CSV] Reading metadata from: {csv_file_path}")
    t0 = time.perf_counter()
    
    # Chunked read to map place_ids to rich hierarchy (Major, Mid, Sub)
    cols_to_read = ['상가업소번호', '상호명', '지점명', '상권업종대분류명', '상권업종중분류명', '상권업종소분류명']
    mapping = {}
    chunk_size = 50000
    for chunk in pd.read_csv(csv_file_path, usecols=cols_to_read, dtype=str, encoding='utf-8', chunksize=chunk_size):
        chunk_filtered = chunk[chunk['상가업소번호'].isin(existing_place_ids)]
        for _, row in chunk_filtered.iterrows():
            mapping[row['상가업소번호']] = {
                "branch": str(row['지점명']).strip() if pd.notna(row['지점명']) else "",
                "major": str(row['상권업종대분류명']).strip() if pd.notna(row['상권업종대분류명']) else "",
                "mid": str(row['상권업종중분류명']).strip() if pd.notna(row['상권업종중분류명']) else "",
                "sub": str(row['상권업종소분류명']).strip() if pd.notna(row['상권업종소분류명']) else ""
            }
            
    print(f"[CSV] Parsed metadata for {len(mapping)} records in {time.perf_counter() - t0:.2f} seconds.")
    
    # 4. Generate V3 Descriptions
    print("[MIGRATE] Generating V3 rich context descriptions...")
    updated_records = []
    for row in existing_rows:
        db_id, place_id, name, cat, address = row
        meta = mapping.get(place_id, {"branch": "", "major": "", "mid": "", "sub": cat})
        
        full_name = f"{name} {meta['branch']}".strip()
        
        desc_v3 = get_semantic_desc(full_name, meta['major'], meta['mid'], meta['sub'])
        updated_records.append({
            "id": db_id,
            "desc_v3": desc_v3
        })
        
    # Write V3 descriptions to PostgreSQL
    print("[DB] Updating 'embedding_text_v3' in database...")
    t_update = time.perf_counter()
    with engine.begin() as conn:
        for r in updated_records:
            conn.execute(
                text("UPDATE places SET embedding_text_v3 = :desc WHERE id = :id"),
                {"desc": r["desc_v3"], "id": r["id"]}
            )
    print(f"[DB] Updated embedding_text_v3 for {len(updated_records)} rows in {time.perf_counter() - t_update:.2f} seconds.")
    
    # 5. Compute V3 Embeddings
    print(f"[MODEL] Loading '{MODEL_NAME}' on {DEVICE}...", flush=True)
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    
    # Query only rows that have embedding_text_v3 set, but lack embedding_vector_v3 (for idempotent resume!)
    print("[DB] Checking rows requiring V3 embedding calculation...", flush=True)
    with engine.connect() as conn:
        rows_to_compute = conn.execute(text(
            "SELECT id, embedding_text_v3 FROM places WHERE embedding_text_v3 IS NOT NULL AND embedding_vector_v3 IS NULL"
        )).all()
        
    total_to_compute = len(rows_to_compute)
    print(f"[DB] Found {total_to_compute} rows remaining to embed out of {len(updated_records)} total rows.", flush=True)
    
    if total_to_compute > 0:
        print("[MODEL] Beginning V3 embedding calculations in batches...", flush=True)
        t_emb_start = time.perf_counter()
        
        batch_size = 200
        completed = 0
        
        for i in range(0, total_to_compute, batch_size):
            batch = rows_to_compute[i:i+batch_size]
            batch_ids = [r[0] for r in batch]
            batch_texts = [r[1] for r in batch]
            
            t_batch = time.perf_counter()
            with torch.no_grad():
                batch_embs = model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).tolist()
                
            # Immediately save this batch to the database
            with engine.begin() as conn:
                for db_id, emb in zip(batch_ids, batch_embs):
                    conn.execute(
                        text("UPDATE places SET embedding_vector_v3 = :emb WHERE id = :db_id"),
                        {"emb": emb, "db_id": db_id}
                    )
            
            completed += len(batch)
            elapsed = time.perf_counter() - t_emb_start
            batch_time = time.perf_counter() - t_batch
            percent = (completed / total_to_compute) * 100
            print(f"[MODEL PROGRESS] Embedded {completed}/{total_to_compute} ({percent:.1f}%) | "
                  f"Batch: {batch_time:.2f}s | Total Elapsed: {elapsed:.1f}s | Remaining Est: {(elapsed / completed) * (total_to_compute - completed):.1f}s", flush=True)
                  
        print(f"[MODEL] All V3 embeddings computed and stored successfully in {time.perf_counter() - t_emb_start:.2f} seconds!", flush=True)
    else:
        print("[DB] All active rows already have V3 embeddings computed. Skipping computation.", flush=True)
        
    print("\n[SUCCESS] V3 Migration & Embedding Generation Completed Successfully!", flush=True)
    print("="*80, flush=True)

if __name__ == "__main__":
    main()
