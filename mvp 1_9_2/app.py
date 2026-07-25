from flask import Flask, request, jsonify, render_template, send_file
import threading, os, uuid, importlib

# 분석 엔진 선택 (환경변수 ENGINE_VERSION 로 전환, 기본 1_9_2)
#   현재는 mvp_ver_1_9_2 단일 엔진. 추후 버전 추가 시 아래 맵에 등록하면 전환 가능.
_ENGINE_MODULES = {
    "1_9_2":         "mvp_ver_1_9_2",
    "mvp_ver_1_9_2": "mvp_ver_1_9_2",
}
_DEFAULT_ENGINE = "mvp_ver_1_9_2"
_engine_key  = os.getenv("ENGINE_VERSION", "1_9_2")
_engine_name = _ENGINE_MODULES.get(_engine_key, _DEFAULT_ENGINE)
_engine = importlib.import_module(_engine_name)
print(f"🚀 분석 엔진 로드: {_engine_name} (ENGINE_VERSION={_engine_key})")

CrisisConsultantSystem = _engine.CrisisConsultantSystem
is_youtube_url         = _engine.is_youtube_url
is_google_drive_url    = _engine.is_google_drive_url

app = Flask(__name__)

# 시스템 초기화 (서버 시작 시 한 번만)
system = CrisisConsultantSystem(
    db_path            = 'case_db.json',
    rules_yaml         = 'rules.yaml',
    labels_yaml        = 'controversy_labels.yaml',
    worst_actions_yaml = 'worst_actions_map.yaml',
    template_path      = 'report template.md',
    report_dir         = 'reports',
)

# 분석 작업 상태 저장 (job_id → 상태/결과)
jobs = {}

def run_analysis(job_id, user_input):
    try:
        jobs[job_id]['status'] = 'running'

        if is_youtube_url(user_input) or is_google_drive_url(user_input) or os.path.exists(user_input):
            report, json_path, md_path, pdf_path = system.analyze_video_full(
                video_input=user_input,
                output_dir='samples/transcripts',
                download_dir='downloads',
                use_cache=True,
            )
            jobs[job_id]['md_path']  = md_path
            jobs[job_id]['pdf_path'] = pdf_path
        else:
            # 텍스트 입력
            cases, dists, summary = system.search_and_analyze(user_input)
            classification = system.classify_controversy(user_input)
            report = {
                'meta': {'input_query': user_input},
                'classification': classification,
                'similar_cases': [{'title': c['title']} for c in cases],
                'pattern_summary': summary,
            }

        jobs[job_id]['status'] = 'done'
        jobs[job_id]['report'] = report

    except Exception as e:
        jobs[job_id]['status'] = 'error'
        jobs[job_id]['error']  = str(e)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.json.get('input', '').strip()
    if not user_input:
        return jsonify({'error': '입력값이 없습니다'}), 400

    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'queued'}

    thread = threading.Thread(target=run_analysis, args=(job_id, user_input))
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({'error': '없는 작업입니다'}), 404
    return jsonify(job)


@app.route('/download/<job_id>/<file_type>')
def download(job_id, file_type):
    job = jobs.get(job_id, {})
    path = job.get(f'{file_type}_path')
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': '파일 없음'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)