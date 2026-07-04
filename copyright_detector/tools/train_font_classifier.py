"""
tools/train_font_classifier.py - 폰트 분류 CNN 학습
사용법:
  python tools/train_font_classifier.py                 # 시스템 폰트 + fonts_train/ 학습
  python tools/train_font_classifier.py --samples 400 --epochs 10

학습 데이터:
  1. 시스템 내장 폰트 (아래 SYSTEM_FONTS — HY 상업 폰트 + 무료/번들 폰트)
  2. fonts_train/ 폴더의 .ttf/.otf — manifest.json 으로 상업 여부 지정
     (구매한 상업 폰트 파일을 넣고 재학습하면 감지 범위 확장)

fonts_train/manifest.json 형식:
  [{"file": "격동고딕.otf", "name": "격동고딕", "commercial": true,
    "foundry": "Sandoll", "risk": 0.88}]
  manifest에 없는 파일은 파일명을 클래스명으로, commercial=true 로 간주.

설계: 상업 폰트만 학습하면 모든 자막이 상업 폰트로 강제 분류되므로
무료/시스템 폰트를 거절(negative) 클래스로 반드시 포함한다.
"""
import os
import sys
import json
import time
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from utils.font_classifier import (
    render_text_sample, build_model, MODEL_DIR, MODEL_PATH, META_PATH,
    INPUT_H, INPUT_W,
)

WIN_FONTS = r"C:\Windows\Fonts"
USER_FONTS = os.path.expanduser(r"~\AppData\Local\Microsoft\Windows\Fonts")

# ─────────────────────────────────────────────
# 시스템 폰트 매니페스트
# (file, index) 목록을 같은 클래스로 묶음 — 여러 굵기 = 한 폰트
# commercial: 상업 폰트 여부 (True만 finding 대상)
# 굴림/바탕 등 Windows 번들은 사실상 전 사용자 라이선스 보유 → negative
# ─────────────────────────────────────────────
SYSTEM_FONTS = [
    # ── HY 상업 폰트 (한양정보통신 — Office/HWP 번들이지만 상업 파운드리) ──
    {"name": "HY헤드라인M",  "files": [("H2HDRM.TTF", 0)], "commercial": True,  "foundry": "HY", "risk": 0.82},
    {"name": "HY중고딕",     "files": [("H2GTRM.TTF", 0), ("H2GTRE.TTF", 0)], "commercial": True, "foundry": "HY", "risk": 0.82},
    {"name": "HY신명조",     "files": [("H2MJRE.TTF", 0), ("H2MJSM.TTF", 0)], "commercial": True, "foundry": "HY", "risk": 0.82},
    {"name": "HY그래픽",     "files": [("H2GPRM.TTF", 0)], "commercial": True,  "foundry": "HY", "risk": 0.77},
    {"name": "HY엽서",       "files": [("H2PORL.TTF", 0), ("H2PORM.TTF", 0)], "commercial": True, "foundry": "HY", "risk": 0.77},
    {"name": "HY샘물M",      "files": [("H2SA1M.TTF", 0)], "commercial": True,  "foundry": "HY", "risk": 0.72},
    {"name": "HY궁서B",      "files": [("H2GSRB.TTF", 0)], "commercial": True,  "foundry": "HY", "risk": 0.72},
    {"name": "HY목각파임B",  "files": [("H2MKPB.TTF", 0)], "commercial": True,  "foundry": "HY", "risk": 0.82},

    # ── Windows 번들 (negative — 전 사용자 사실상 보유) ──
    {"name": "굴림",     "files": [("gulim.ttc", 0), ("gulim.ttc", 1)],  "commercial": False, "foundry": "HY/Windows"},
    {"name": "돋움",     "files": [("gulim.ttc", 2), ("gulim.ttc", 3)],  "commercial": False, "foundry": "HY/Windows"},
    {"name": "바탕",     "files": [("batang.ttc", 0), ("batang.ttc", 1)], "commercial": False, "foundry": "HY/Windows"},
    {"name": "궁서",     "files": [("batang.ttc", 2), ("batang.ttc", 3)], "commercial": False, "foundry": "HY/Windows"},
    {"name": "새굴림",   "files": [("NGULIM.TTF", 0)], "commercial": False, "foundry": "HY/Windows"},
    {"name": "맑은고딕", "files": [("malgun.ttf", 0), ("malgunbd.ttf", 0), ("malgunsl.ttf", 0)],
     "commercial": False, "foundry": "Microsoft"},

    # ── 무료 폰트 (negative) ──
    {"name": "본고딕(Noto Sans KR)",  "files": [("NotoSansKR-VF.ttf", 0)],  "commercial": False, "foundry": "Google"},
    {"name": "본명조(Noto Serif KR)", "files": [("NotoSerifKR-VF.ttf", 0)], "commercial": False, "foundry": "Google"},
    {"name": "배민주아체",  "files": [("BMJUA_otf.otf", 0)],     "commercial": False, "foundry": "Woowa Bros"},
    {"name": "배민한나체",  "files": [("BMHANNAProOTF.otf", 0)], "commercial": False, "foundry": "Woowa Bros"},
    {"name": "KIMM체",     "files": [("KIMM_bold.ttf", 0), ("KIMM_Light.ttf", 0)], "commercial": False, "foundry": "한국기계연구원"},
    {"name": "에이투지체", "files": [("에이투지체-4Regular.ttf", 0), ("에이투지체-7Bold.ttf", 0), ("에이투지체-9Black.ttf", 0)],
     "commercial": False, "foundry": "에이투지"},
]


def _resolve(fname: str) -> str:
    for d in (WIN_FONTS, USER_FONTS):
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return ""


def collect_classes(fonts_dir: str) -> list:
    """시스템 매니페스트 + fonts_train/ 사용자 폰트 → 학습 클래스 목록."""
    classes = []
    for entry in SYSTEM_FONTS:
        files = [(_resolve(f), i) for f, i in entry["files"]]
        files = [(p, i) for p, i in files if p]
        if files:
            classes.append({**entry, "files": files})
        else:
            print(f"  [스킵] {entry['name']} — 폰트 파일 없음")

    # 사용자 추가 폰트 (구매한 상업 폰트를 fonts_train/에 넣으면 자동 포함)
    if os.path.isdir(fonts_dir):
        manifest = {}
        mpath = os.path.join(fonts_dir, "manifest.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                manifest = {m["file"]: m for m in json.load(f)}
        for fname in sorted(os.listdir(fonts_dir)):
            if not fname.lower().endswith((".ttf", ".otf", ".ttc")):
                continue
            m = manifest.get(fname, {})
            classes.append({
                "name":       m.get("name", os.path.splitext(fname)[0]),
                "files":      [(os.path.join(fonts_dir, fname), int(m.get("index", 0)))],
                "commercial": bool(m.get("commercial", True)),
                "foundry":    m.get("foundry", ""),
                "risk":       float(m.get("risk", 0.80)),
            })
    return classes


def build_dataset(classes: list, n_per_class: int, seed: int = 7):
    rng = random.Random(seed)
    X, y = [], []
    for ci, cls in enumerate(classes):
        made = 0
        attempts = 0
        while made < n_per_class and attempts < n_per_class * 3:
            attempts += 1
            path, idx = rng.choice(cls["files"])
            arr = render_text_sample(path, idx, rng=rng, augment=True)
            if arr is None:
                continue
            X.append(arr)
            y.append(ci)
            made += 1
        print(f"  [{ci:2d}] {cls['name']:<22} {made}샘플"
              + (" (상업)" if cls["commercial"] else ""))
        if made < n_per_class // 2:
            print(f"       ⚠ 렌더링 실패 다수 — 한글 미지원 폰트일 수 있음")
    return np.stack(X), np.array(y, dtype=np.int64)


def train(args):
    import torch
    import torch.nn as nn

    print("── 클래스 수집 ──")
    classes = collect_classes(args.fonts_dir)
    n_comm = sum(1 for c in classes if c["commercial"])
    print(f"총 {len(classes)}개 클래스 (상업 {n_comm} / 무료·번들 {len(classes) - n_comm})\n")
    if n_comm == 0:
        print("⚠ 상업 폰트 클래스가 없습니다 — 학습 의미 없음")
        return

    print(f"── 합성 샘플 렌더링 ({args.samples}/클래스) ──")
    t0 = time.time()
    X, y = build_dataset(classes, args.samples)
    print(f"렌더링 완료: {len(X)}샘플, {time.time()-t0:.0f}초\n")

    # train/val 분할
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(X))
    n_val = max(int(len(X) * 0.12), len(classes))
    vi, ti = idx[:n_val], idx[n_val:]
    Xt, yt, Xv, yv = X[ti], y[ti], X[vi], y[vi]

    model = build_model(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()

    def to_tensor(a):
        t = torch.from_numpy(a)[:, None, :, :]
        return (t - 0.5) / 0.5

    print(f"── 학습 ({args.epochs} epochs, train {len(Xt)} / val {len(Xv)}) ──")
    best_acc = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(Xt))
        tot_loss = 0.0
        for s in range(0, len(perm), args.batch):
            b = perm[s:s + args.batch]
            xb, yb = to_tensor(Xt[b]), torch.from_numpy(yt[b])
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(b)
        sched.step()

        # 검증
        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(Xv), 256):
                xb = to_tensor(Xv[s:s + 256])
                pred = model(xb).argmax(dim=-1).numpy()
                correct += int((pred == yv[s:s + 256]).sum())
        acc = correct / len(Xv)
        print(f"  epoch {ep+1:2d}/{args.epochs}  loss={tot_loss/len(Xt):.4f}  val_acc={acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    # 메타 저장
    meta = {
        "input": {"h": INPUT_H, "w": INPUT_W},
        "val_accuracy": round(best_acc, 4),
        "classes": [
            {"name": c["name"], "commercial": c["commercial"],
             "foundry": c.get("foundry", ""), "risk": c.get("risk", 0.5)}
            for c in classes
        ],
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 저장: {MODEL_PATH} (best val_acc={best_acc:.3f})")
    print(f"   메타: {META_PATH}")

    # 상업 폰트별 검증 정확도 리포트
    import collections
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    per = collections.defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for s in range(0, len(Xv), 256):
            pred = model(to_tensor(Xv[s:s + 256])).argmax(dim=-1).numpy()
            for p_, t_ in zip(pred, yv[s:s + 256]):
                per[int(t_)][1] += 1
                if p_ == t_:
                    per[int(t_)][0] += 1
    print("\n── 클래스별 검증 정확도 ──")
    for ci, (c_, n_) in sorted(per.items()):
        tag = "상업" if classes[ci]["commercial"] else "무료"
        print(f"  [{tag}] {classes[ci]['name']:<22} {c_}/{n_} ({c_/max(n_,1):.0%})")

    # ── 실제로 중요한 지표: 상업 vs 무료 이진 정확도 + finding 게이트 적용 시 ──
    # (사용자 목적 = "상업 폰트인가 아닌가" + conf≥0.80·margin≥0.25 게이트)
    comm_flag = np.array([1 if c["commercial"] else 0 for c in classes])
    bin_correct = bin_total = 0
    gated_fp = gated_tp = gated_total_comm = 0
    with torch.no_grad():
        for s in range(0, len(Xv), 256):
            logits = model(to_tensor(Xv[s:s + 256]))
            probs = torch.softmax(logits, dim=-1).numpy()
            for row, t_ in zip(probs, yv[s:s + 256]):
                order = np.argsort(row)[::-1]
                p1 = int(order[0])
                conf = float(row[p1]); margin = conf - float(row[int(order[1])])
                # 이진: 예측 클래스의 상업여부 == 정답 클래스의 상업여부
                bin_total += 1
                if comm_flag[p1] == comm_flag[int(t_)]:
                    bin_correct += 1
                # 게이트 통과 시 상업 판정의 정/오탐
                fires = comm_flag[p1] == 1 and conf >= 0.80 and margin >= 0.25
                if comm_flag[int(t_)] == 1:
                    gated_total_comm += 1
                    if fires:
                        gated_tp += 1
                elif fires:
                    gated_fp += 1   # 무료인데 상업으로 발화 = 오탐
    print("\n── 상업/무료 이진 정확도 (실사용 핵심 지표) ──")
    print(f"  이진 정확도: {bin_correct}/{bin_total} ({bin_correct/max(bin_total,1):.1%})")
    print(f"  게이트(conf≥0.80,margin≥0.25) 적용 시:")
    print(f"    상업 폰트 탐지율(recall): {gated_tp}/{gated_total_comm} "
          f"({gated_tp/max(gated_total_comm,1):.1%})")
    print(f"    무료 폰트 오탐(false positive): {gated_fp}건 "
          f"(낮을수록 좋음 — 0이 이상적)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=300, help="클래스당 샘플 수")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--fonts-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts_train"))
    train(ap.parse_args())
