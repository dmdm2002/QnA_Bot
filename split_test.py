import re
from kss import split_sentences
from kiwipiepy import Kiwi

text = """교통사고 5대 과실은 교통사고 발생 시 책임을 판단하는 기준으로 사용되는 개념입니다. 다섯 가지 과실은 다음과 같습니다:

1. 운전자 과실: 운전자가 교통규칙을 위반하거나 운전 중 부주의로 인해 사고가 발생한 경우
2. 보행자 과실: 보행자가 횡단보도를 건널 때 신호를 무시하거나 안전거리를 유지하지 않아 사고가 발생한 경우
3. 차량 과실: 차량의 기능적 결함이나 정비 미흡으로 인해 사고가 발생한 경우
4. 도로 과실: 도로의 설계 미흡이나 유지보수 부족으로 인해 사고가 발생한 경우
5. 기타 과실: 위의 네 가지 과실 외의 다른 요인으로 인해 사고가 발생한 경우

이러한 5대 과실은 교통사고 발생 시 책임을 명확히 하기 위해 사용되며, 각각의 상황에 따라 적용되는 내용이 달라질 수 있습니다.
"""

text_list = text.split('\n')

split_text_list = []
for split_text in text_list:
    split_text = split_sentences(split_text)
    split_text_list += split_text

print(split_text_list)

# text = re.compile('\n').sub(' ', text)
# print(text)
#
# print(split_sentences(text))
#
# kiwi = Kiwi()
# print(kiwi.split_into_sents(text))