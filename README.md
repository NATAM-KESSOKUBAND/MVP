reports 파일에 이제까지 돌려본 결과들 있음

api 돌리면 로컬로 웹사이트에서 실행 가능, pdf 리포트 생성 가능

지금은 mvp 1_9_2.py 로 문장 분석 및 PRS CRS 판단

곧 copyright detector도 합쳐서 최종본 MVP 만들 예정

controversy_samples.json과 cade_db.json을 이용해서 controversy_model.joblib를 classifier_model.py로 문장 학습시킴

trend_updater.py를 사용해서 최근 뉴스/기사 분석, 키워드, 사건 등을 찾고 review_update.py로 검토 후 controversy_model.joblib 학습