tui.py 실행



-----------------예전-----------------
실행 방법: python main.py "영상파일명.mp4"

다시 처음부터 강제로 시작: python main.py "영상파일명.mp4" --force


영상 다운로드:

python utils/downloader.py "https://youtu.be/xxxx"
저장 폴더 지정:

python utils/downloader.py "https://youtu.be/xxxx" --output ./downloads
화질 선택 (기본 1080p):

python utils/downloader.py "https://youtu.be/xxxx" --quality 720
정보만 보기 (다운로드 없이):

python utils/downloader.py "https://youtu.be/xxxx" --info
다운로드 후 바로 분석:

python utils/downloader.py "https://youtu.be/xxxx"
python main.py "downloads/영상제목.mp4"
유튜브 외에도 인스타그램, 트위터/X, 틱톡 등 yt-dlp가 지원하는 1000개 이상 사이트에서 다 됩니다.