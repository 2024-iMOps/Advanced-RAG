def LLMPrompt(): 
    prompt = """You are an assistant for question-answering tasks.

Use the following pieces of retrieved context to answer the question. \
Provide an answer between one to three sentences and keep it concise. \
Answer only the question being asked. \
Write the answer in Korean.

context: {context}

question: {question}

answer:
"""
    return prompt


def IntentPrompt():
    prompt = """당신은 사용자 질문의 의도를 분석하는 전문가입니다. \

사용자의 질문이 은행 상품과 관련있는지 여부를 판단하고, \
관련이 있다면 yes, 없다면 no를 반환해주세요. \

[Example]
if 강남역 맛집 추천해줘.:
    return 'no'
elif 직불카드는 할부결제가 가능한가요?:
    return 'yes'
elif 오늘 날씨 너무 좋다.:
    return 'no'
elif A회사의 투자 위험성이 너무 높아.:
    return 'yes'

question: {question}

answer:
"""
    return prompt


def IntentNOPrompt():
    prompt = """이 질문은 은행 업무와 관련이 없습니다. \
사용자에게 은행 업무와 관련이 없다는 것을 알려주세요.

question: {question}

answer:
"""
    return prompt


def EvaluationPrompt():
    prompt = """
작업:
주어진 질문과 답변 페어를 평가 기준에 따라 평가해주세요.

평가 기준:
각 항목마다 10점 만점으로 평가해주세요.

1. accuracy: 질문을 정확히 이해하고 관련된 정보를 제공했는지 평가합니다.
2. conciseness: 불필요한 내용 없이 핵심 정보를 효율적으로 전달했는지 평가합니다.

평가 결과 형식:
- accuracy: X/10점
- conciseness: X/10점
- feedback: 해당 점수를 부여한 이유를 상세히 서술하세요.

예시:

Question: 해외 서비스의 수수료율은 어떻게 되나요?
Answer: 해외 서비스의 수수료율은 0.25% 입니다.
accuracy: 10/10점
conciseness: 10/10점
feedback: 답변이 질문을 정확히 이해하고 해외 서비스의 수수료율을 정확하게 제공했습니다. 불필요한 내용 없이 핵심 정보를 간결하게 전달하여 매우 우수합니다.

Question: 아이엠뱅크 고객센터 전화번호를 알려줘.
Answer: 아이엠뱅크 고객센터 전화번호는 1566-5050입니다. 추가 문의 사항이 있으시면 언제든지 연락주시기 바랍니다. 저희는 고객님의 만족을 위해 최선을 다하고 있습니다.
accuracy: 10/10점
conciseness: 7/10점
feedback: 질문에 대한 정확한 답변을 제공했지만, 추가적인 문구가 있어 불필요한 내용이 포함되었습니다. 핵심 정보만 전달했다면 더욱 간결한 답변이 되었을 것입니다.

Question: 자동 납부 업무 시간이 지나게 되면 결제를 어떻게 해?
Answer: 자동 납부 업무 시간이 지나면 결제가 불가능하며, 다음 영업일까지 기다리셔야 합니다.
accuracy: 5/10점
conciseness: 9/10점
feedback: 질문을 이해했지만 제공된 정보가 실제와 다릅니다. 실제로는 자동 납부 시간이 지나도 다른 방법으로 당일 결제가 가능합니다. 답변은 간결하지만 정확한 정보를 전달하지 못했습니다.

Question: 보험계약자는 청약을 언제까지 철회할 수 있습니까?
Answer: 보험계약자는 청약을 한 날부터 15일 이내에 철회할 수 있습니다. 단, 진단계약이나 단체(취급) 계약의 경우에는 예외가 있으며, 통신판매 계약의 경우에는 30일 이내에 철회가 가능합니다. 자세한 사항은 보험 약관을 참조하시기 바랍니다.
accuracy: 9/10점
conciseness: 8/10점
feedback: 답변이 질문에 대한 거의 모든 정보를 제공했지만, 약간의 추가 정보로 인해 간결성이 떨어집니다. 그러나 필요한 예외 사항을 포함하여 정확성은 높습니다.

Question: 금융감독원 전화번호와 홈페이지를 알려주세요.
Answer: 금융감독원의 전화번호는 국번 없이 1332이고, 홈페이지는 www.fss.or.kr 입니다.
accuracy: 10/10점
conciseness: 10/10점
feedback: 질문에 대한 답변을 정확하고 완벽하게 제공했습니다. 불필요한 내용 없이 필요한 정보만을 간결하게 전달하여 매우 훌륭합니다.

결과를 다음 JSON 형식으로 반환해 주세요:
{
  "accuracy": 점수,
  "conciseness": 점수,
  "feedback": "피드백 내용"
}

"""

    return prompt