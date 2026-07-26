from flask import Flask, request, jsonify, render_template, send_file
import threading, os, uuid, time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)

import mvp_ver_1_9_3 as v193

mvp = v193.mvp  # v1.9.2 엔진 (v193 내부에서 재사용 중인 모듈)

app = Flask(__name__, template_folder=str(SCRIPT_DIR.parent / "templates"))

# 시스템 초기화 (서버 시작 시 한 번만)
system = mvp.CrisisConsultantSystem(
    db_path            = 'case_db.json',
    rules_yaml         = 'rules.yaml',
    labels_yaml        = 'controversy_labels.yaml',
    worst_actions_yaml = 'worst_actions_map.yaml',
    template_path      = 'report template.md',
    report_dir         = 'reports',
)

# 분석 작업 상태 저장 (job_id → 상태/결과). 여러 요청 스레드가 동시에
# 접근하므로 락으로 보호하고, 오래된 작업은 새 작업 생성 시 정리한다.
JOB_TTL_SECONDS = 3600
_jobs_lock = threading.Lock()
jobs = {}


def _prune_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    stale = [jid for jid, j in jobs.items() if j.get('created_at', 0) < cutoff]
    for jid in stale:
        jobs.pop(jid, None)


def _update_job(job_id, **fields):
    with _jobs_lock:
        jobs[job_id].update(fields)


def run_analysis(job_id, user_input):
    try:
        _update_job(job_id, status='running')

        if mvp.is_youtube_url(user_input):
            report, json_path = v193.analyze_only(system, user_input, download_dir='downloads')
            local_video = (report.get('youtube_meta') or {}).get('video_path')
            copyright_results = v193.run_copyright_detection(local_video) if local_video else None
            md_path, pdf_path = v193.finalize_reports(report, json_path, copyright_results)
            _update_job(job_id, md_path=md_path, pdf_path=pdf_path)

        elif mvp.is_google_drive_url(user_input):
            dl = mvp.download_google_drive_video(user_input, 'downloads')
            local_video = dl['video_path']
            report, json_path = v193.analyze_only(system, local_video)
            copyright_results = v193.run_copyright_detection(local_video)
            md_path, pdf_path = v193.finalize_reports(report, json_path, copyright_results)
            _update_job(job_id, md_path=md_path, pdf_path=pdf_path)

        elif os.path.exists(user_input):
            report, json_path = v193.analyze_only(system, user_input)
            copyright_results = v193.run_copyright_detection(user_input)
            md_path, pdf_path = v193.finalize_reports(report, json_path, copyright_results)
            _update_job(job_id, md_path=md_path, pdf_path=pdf_path)

        else:
            # 텍스트 입력 → 영상이 없어 저작권 분석은 수행하지 않고 NATAM 분석만 수행
            cases, dists, summary = system.search_and_analyze(user_input)
            classification = system.classify_controversy(user_input)
            report = {
                'meta': {'input_query': user_input},
                'classification': classification,
                'similar_cases': [{'title': c['title']} for c in cases],
                'pattern_summary': summary,
                'copyright': None,
            }

        _update_job(job_id, status='done', report=report)

    except Exception as e:
        _update_job(job_id, status='error', error=str(e))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    user_input = request.json.get('input', '').strip()
    if not user_input:
        return jsonify({'error': '입력값이 없습니다'}), 400

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _prune_old_jobs()
        jobs[job_id] = {'status': 'queued', 'created_at': time.time()}

    thread = threading.Thread(target=run_analysis, args=(job_id, user_input))
    thread.start()

    return jsonify({'job_id': job_id})


@app.route('/status/<job_id>')
def status(job_id):
    with _jobs_lock:
        job = dict(jobs[job_id]) if job_id in jobs else None
    if not job:
        return jsonify({'error': '없는 작업입니다'}), 404
    return jsonify(job)


@app.route('/download/<job_id>/<file_type>')
def download(job_id, file_type):
    with _jobs_lock:
        job = dict(jobs.get(job_id, {}))
    path = job.get(f'{file_type}_path')
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({'error': '파일 없음'}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
