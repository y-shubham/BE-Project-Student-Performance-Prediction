import re
import io
import math
import zipfile

import pandas as pd
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_file
from flask_cors import CORS
import joblib
from sklearn.model_selection import train_test_split
from werkzeug.exceptions import NotFound, Forbidden
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt
)

# ---- PDF / plotting deps ----
import matplotlib
matplotlib.use("Agg")  # headless for servers
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader  # <-- IMPORTANT for drawImage on BytesIO

app = Flask(__name__)
CORS(app)
app.config["JWT_SECRET_KEY"] = "change-me-in-prod"
jwt = JWTManager(app)

# ---------- Load model & data ----------
model = joblib.load("student_model.pkl")
data = pd.read_csv("Students_Performance_Dataset.csv")
assert "Final_Score" in data.columns, "Dataset must have Final_Score column"

# ---------- Helper: mentors from departments (+ optional overrides) ----------
def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

unique_depts = sorted(data["Department"].dropna().unique().tolist())

# Optionally map real mentor names by department: {"Department Name": "Real Mentor Name"}
MENTOR_NAME_OVERRIDES = {}

MENTORS = []
for dept in unique_depts:
    mname = MENTOR_NAME_OVERRIDES.get(dept, f"{dept} Mentor")
    mid = slug(dept)
    MENTORS.append({"mentor_id": mid, "mentor_name": mname, "department": dept})

MENTOR_BY_ID = {m["mentor_id"]: m for m in MENTORS}

# ---------- Cached model score ----------
try:
    X = data.drop(columns=["Final_Score"])
    y = data["Final_Score"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    MODEL_R2 = float(model.score(X_test, y_test))
except Exception:
    MODEL_R2 = 0.0

# ---------- JWT helpers ----------
def make_claims(role, **extra):
    claims = {"role": role}
    claims.update(extra)
    return claims

def require_role(allowed):
    def wrapper(fn):
        from functools import wraps
        @wraps(fn)
        def inner(*args, **kwargs):
            claims = get_jwt()
            role = claims.get("role")
            if role not in allowed:
                raise Forbidden("Not allowed")
            return fn(*args, **kwargs)
        return inner
    return wrapper

def _check_access_student(student_id, claims):
    role = claims.get("role")
    if role == "student" and claims.get("student_id") != student_id:
        raise Forbidden("Students can only view their own data")
    if role == "mentor":
        dept = claims.get("department")
        if data.loc[(data["Student_ID"] == student_id) & (data["Department"] == dept)].empty:
            raise Forbidden("Mentors can only view students in their department")

# ---------- Pages ----------
@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/")
@jwt_required(optional=True)
def home():
    if get_jwt() is None:
        return redirect(url_for("login_page"))
    return render_template("index.html")

@app.route("/mentor/<mentor_id>")
@jwt_required(optional=True)
def mentor_page(mentor_id):
    if mentor_id not in MENTOR_BY_ID:
        raise NotFound("Mentor not found")
    claims = get_jwt()
    if claims is None:
        return redirect(url_for("login_page"))
    role = claims.get("role")
    if role == "mentor" and claims.get("mentor_id") != mentor_id:
        raise Forbidden("Mentors can only view their department")
    return render_template("mentor.html")

# ---------- Auth API ----------
@app.route("/login", methods=["POST"])
def login_api():
    payload = request.get_json(force=True)
    mode = payload.get("mode")

    if mode == "mentor":
        username = payload.get("username", "")
        password = payload.get("password", "")
        if not username.startswith("mentor-"):
            return jsonify({"error": "Invalid mentor username"}), 401

        mentor_id = username[len("mentor-"):]
        if mentor_id not in MENTOR_BY_ID:
            return jsonify({"error": "Unknown mentor"}), 401

        dept = MENTOR_BY_ID[mentor_id]["department"]
        expected_pw = f"{dept}123"   # DepartmentName + 123

        if password != expected_pw:
            return jsonify({"error": "Invalid mentor credentials"}), 401

        token = create_access_token(
            identity=username,
            additional_claims=make_claims(
                "mentor",
                mentor_id=mentor_id,
                department=dept,
            ),
        )
        return jsonify({"access_token": token, "role": "mentor"})

    elif mode == "student":
        student_id = payload.get("student_id")
        password = payload.get("password", "")
        if student_id is None or data.loc[data["Student_ID"] == student_id].empty:
            return jsonify({"error": "Unknown student ID"}), 401

        expected_pw = f"hello{student_id}"   # hello + Student_ID
        if password != expected_pw:
            return jsonify({"error": "Invalid student credentials"}), 401

        row = data.loc[data["Student_ID"] == student_id].iloc[0]
        token = create_access_token(
            identity=student_id,
            additional_claims=make_claims(
                "student",
                student_id=student_id,
                department=row["Department"],
            ),
        )
        return jsonify({"access_token": token, "role": "student"})

    elif mode == "admin":
        if payload.get("username") == "admin" and payload.get("password") == "admin123":
            token = create_access_token(identity="admin", additional_claims=make_claims("admin"))
            return jsonify({"access_token": token, "role": "admin"})
        return jsonify({"error": "Invalid admin credentials"}), 401

    else:
        return jsonify({"error": "Invalid mode"}), 400

# ---------- Data APIs (secured) ----------
@app.route("/api/model_accuracy")
@jwt_required()
def api_model_accuracy():
    return jsonify({"r_squared": MODEL_R2})

@app.route("/api/mentors")
@jwt_required()
def api_mentors():
    # Intentionally visible to admin and mentors (admin uses it on /, mentor page uses it to resolve header)
    return jsonify(MENTORS)

@app.route("/api/students")
@jwt_required()
def api_students():
    claims = get_jwt()
    role = claims.get("role")
    if role == "admin":
        rows = data[["Student_ID", "First_Name", "Last_Name", "Department"]]
    elif role == "mentor":
        dept = claims.get("department")
        rows = data.loc[data["Department"] == dept, ["Student_ID", "First_Name", "Last_Name", "Department"]]
    else:
        sid = claims.get("student_id")
        rows = data.loc[data["Student_ID"] == sid, ["Student_ID", "First_Name", "Last_Name", "Department"]]
    return jsonify(rows.to_dict(orient="records"))

@app.route("/api/student_data/<student_id>")
@jwt_required()
def api_student(student_id):
    claims = get_jwt()
    role = claims.get("role")
    if role == "student" and claims.get("student_id") != student_id:
        raise Forbidden("Students can only view their own data")
    if role == "mentor":
        dept = claims.get("department")
        if data.loc[(data["Student_ID"] == student_id) & (data["Department"] == dept)].empty:
            raise Forbidden("Mentors can only view students in their department")

    row = data.loc[data["Student_ID"] == student_id]
    if row.empty:
        return jsonify({"error": "Student not found"}), 404

    # Base scores + prediction
    features = row.drop(columns=["Final_Score"])
    pred = float(model.predict(features)[0])
    r = row.iloc[0]

    # Department averages
    dept = r["Department"]
    subset = data[data["Department"] == dept].copy()
    dept_avg = {
        "midterm": float(subset["Midterm_Score"].mean()),
        "assignments": float(subset["Assignments_Avg"].mean()),
        "quizzes": float(subset["Quizzes_Avg"].mean()),
        "projects": float(subset["Projects_Score"].mean()),
        "participation": float(subset["Participation_Score"].mean()),
    }

    # Student metrics
    student_metrics = {
        "midterm": float(r["Midterm_Score"]),
        "assignments": float(r["Assignments_Avg"]),
        "quizzes": float(r["Quizzes_Avg"]),
        "projects": float(r["Projects_Score"]),
        "participation": float(r["Participation_Score"]),
    }

    # Deltas & suggestions
    deltas = {k: round(student_metrics[k] - dept_avg[k], 2) for k in student_metrics}

    def hint(label, delta):
        if delta <= -10:
            return f"{label}: significantly below dept avg ({delta}). Prioritize this area."
        elif delta < 0:
            return f"{label}: slightly below avg ({delta}). Some improvement recommended."
        elif delta >= 10:
            return f"{label}: strong area (+{delta}). Keep it up."
        else:
            return f"{label}: around avg ({'+' if delta>=0 else ''}{delta}). Maintain consistency."

    suggestions = [
        hint("Midterm", deltas["midterm"]),
        hint("Assignments", deltas["assignments"]),
        hint("Quizzes", deltas["quizzes"]),
        hint("Projects", deltas["projects"]),
        hint("Participation", deltas["participation"]),
    ]

    payload = {
        "student_id": r["Student_ID"],
        "full_name": f"{r['First_Name']} {r['Last_Name']}",
        "department": dept,
        "actual_final_score": float(round(r["Final_Score"], 2)),
        "predicted_final_score": float(round(pred, 2)),
        "graph_data": student_metrics,
        "dept_avg": dept_avg,
        "deltas": deltas,
        "suggestions": suggestions,
    }
    return jsonify(payload)

@app.route("/api/mentor/<mentor_id>/students")
@jwt_required()
def api_mentor_students(mentor_id):
    meta = MENTOR_BY_ID.get(mentor_id)
    if not meta:
        return jsonify({"error": "Mentor not found"}), 404
    claims = get_jwt()
    role = claims.get("role")
    if role == "mentor" and claims.get("mentor_id") != mentor_id:
        raise Forbidden("Mentors can only view their department")
    dept = meta["department"]
    rows = (
        data[data["Department"] == dept][["Student_ID", "First_Name", "Last_Name", "Final_Score"]]
        .sort_values("Final_Score", ascending=False)
        .to_dict(orient="records")
    )
    return jsonify({"mentor": meta, "students": rows})

@app.route("/api/mentor/<mentor_id>/summary")
@jwt_required()
def api_mentor_summary(mentor_id):
    meta = MENTOR_BY_ID.get(mentor_id)
    if not meta:
        return jsonify({"error": "Mentor not found"}), 404
    claims = get_jwt()
    role = claims.get("role")
    if role == "mentor" and claims.get("mentor_id") != mentor_id:
        raise Forbidden("Mentors can only view their department")
    dept = meta["department"]
    subset = data[data["Department"] == dept].copy()
    if subset.empty:
        return jsonify({"mentor": meta, "summary": {}})

    preds = model.predict(subset.drop(columns=["Final_Score"]))
    subset["Predicted_Final_Score"] = preds

    avg_actual = float(subset["Final_Score"].mean())
    avg_pred = float(subset["Predicted_Final_Score"].mean())
    count = int(subset.shape[0])

    at_risk = (
        subset.loc[subset["Predicted_Final_Score"] < 60, ["Student_ID", "First_Name", "Last_Name", "Predicted_Final_Score"]]
        .sort_values("Predicted_Final_Score")
        .head(10)
        .to_dict(orient="records")
    )

    bins = [0, 50, 60, 70, 80, 90, 100]
    hist_actual = subset["Final_Score"].value_counts(bins=bins).sort_index()
    hist_pred = subset["Predicted_Final_Score"].value_counts(bins=bins).sort_index()

    summary = {
        "mentor": meta,
        "count": count,
        "avg_actual": round(avg_actual, 2),
        "avg_predicted": round(avg_pred, 2),
        "at_risk": at_risk,
        "histogram": {
            "bins": [str(i) for i in hist_actual.index.astype(str).tolist()],
            "actual": hist_actual.tolist(),
            "predicted": hist_pred.tolist(),
        },
    }
    return jsonify(summary)

@app.route("/api/at_risk")
@jwt_required()
def api_at_risk():
    """
    Return at-risk students based on predicted score.
    - Admin: across all departments
    - Mentor: only their department
    - Student: only themselves
    Query params:
      - threshold: float, default 60
      - limit: int, default 10
    """
    threshold = float(request.args.get("threshold", 60))
    limit = int(request.args.get("limit", 10))

    claims = get_jwt()
    role = claims.get("role")

    # Build predictions for the accessible slice
    df = data.copy()
    df["Predicted_Final_Score"] = model.predict(df.drop(columns=["Final_Score"]))

    if role == "mentor":
        dept = claims.get("department")
        df = df[df["Department"] == dept]
    elif role == "student":
        sid = claims.get("student_id")
        df = df[df["Student_ID"] == sid]
    # admin sees all

    # Filter & shape
    df = df[df["Predicted_Final_Score"] < threshold]
    df = df.sort_values("Predicted_Final_Score").head(limit)

    rows = df[["Student_ID", "First_Name", "Last_Name", "Department", "Predicted_Final_Score"]].to_dict(orient="records")
    return jsonify({
        "threshold": threshold,
        "count": int(len(rows)),
        "students": rows
    })

# ---------- Report helpers ----------
def _radar_png(student_metrics, dept_avg):
    import numpy as np
    labels = ["Midterm", "Assignments", "Quizzes", "Projects", "Participation"]

    s = [student_metrics["midterm"], student_metrics["assignments"], student_metrics["quizzes"],
         student_metrics["projects"], student_metrics["participation"]]
    a = [dept_avg["midterm"], dept_avg["assignments"], dept_avg["quizzes"],
         dept_avg["projects"], dept_avg["participation"]]

    # Close the loop
    angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    s = s + s[:1]
    a = a + a[:1]

    # High-contrast colors for white paper
    BLUE_600 = "#2563eb"     # student line
    SLATE_700 = "#334155"    # dept avg line
    GRID = "#d1d5db"         # grid lines
    TEXT = "#111827"         # axis labels / ticks

    fig = plt.figure(figsize=(5.2, 5.2), facecolor="white")
    ax = plt.subplot(111, polar=True, facecolor="white")
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids([ang * 180 / math.pi for ang in angles[:-1]], labels,
                      color=TEXT, fontsize=10)
    ax.tick_params(colors=TEXT)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 100)
    ax.grid(True, color=GRID, alpha=0.95, linewidth=0.8)
    ax.spines["polar"].set_color(GRID)

    # Dept avg (dashed)
    ax.plot(angles, a, linewidth=2.2, linestyle=(0, (6, 4)),
            color=SLATE_700, alpha=0.95, label="Dept Avg")
    ax.fill(angles, a, color=SLATE_700, alpha=0.14)

    # Student
    ax.plot(angles, s, linewidth=2.6, color=BLUE_600, alpha=0.98, label="Student")
    ax.fill(angles, s, color=BLUE_600, alpha=0.38)

    leg = ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=True)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(1.0)
    for txt in leg.get_texts():
        txt.set_color(TEXT)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _student_report_pdf(student_id):
    row = data.loc[data["Student_ID"] == student_id]
    if row.empty:
        return None
    r = row.iloc[0]

    # prediction + context
    features = row.drop(columns=["Final_Score"])
    pred = float(model.predict(features)[0])
    dept = r["Department"]
    subset = data[data["Department"] == dept].copy()
    dept_avg = {
        "midterm": float(subset["Midterm_Score"].mean()),
        "assignments": float(subset["Assignments_Avg"].mean()),
        "quizzes": float(subset["Quizzes_Avg"].mean()),
        "projects": float(subset["Projects_Score"].mean()),
        "participation": float(subset["Participation_Score"].mean()),
    }
    student_metrics = {
        "midterm": float(r["Midterm_Score"]),
        "assignments": float(r["Assignments_Avg"]),
        "quizzes": float(r["Quizzes_Avg"]),
        "projects": float(r["Projects_Score"]),
        "participation": float(r["Participation_Score"]),
    }
    deltas = {k: round(student_metrics[k] - dept_avg[k], 2) for k in student_metrics}

    def hint(label, delta):
        if delta <= -10:  return f"{label}: significantly below dept avg ({delta}). Prioritize this area."
        if delta < 0:     return f"{label}: slightly below avg ({delta}). Plan small improvements."
        if delta >= 10:   return f"{label}: strong area (+{delta}). Keep it up."
        return f"{label}: around avg ({'+' if delta>=0 else ''}{delta}). Maintain consistency."

    suggestions = [
        hint("Midterm", deltas["midterm"]),
        hint("Assignments", deltas["assignments"]),
        hint("Quizzes", deltas["quizzes"]),
        hint("Projects", deltas["projects"]),
        hint("Participation", deltas["participation"]),
    ]

    # Colors for white page
    TITLE = colors.HexColor("#111827")     # near-black
    SUB   = colors.HexColor("#334155")     # slate-700
    BODY  = colors.HexColor("#1f2937")     # slate-800
    MUTED = colors.HexColor("#475569")     # slate-600
    BLUE  = colors.HexColor("#2563eb")     # blue-600
    LINE  = colors.HexColor("#e5e7eb")     # gray-200 divider
    WHITE = colors.white

    # Radar image (high contrast) -> ImageReader
    radar_png_buf = _radar_png(student_metrics, dept_avg)
    radar_img = ImageReader(radar_png_buf)

    # Compose PDF
    pdf_buf = io.BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=A4)
    W, H = A4

    # Title
    c.setFillColor(TITLE)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, H-2.5*cm, "AI Performance Report")
    c.setFillColor(SUB)
    c.setFont("Helvetica", 11)
    c.drawString(2*cm, H-3.2*cm, "Student performance, context, and guidance")

    # Divider
    c.setStrokeColor(LINE)
    c.setLineWidth(0.7)
    c.line(2*cm, H-3.5*cm, W-2*cm, H-3.5*cm)

    # --- Identity line with white backdrop so it never gets obscured ---
    name_y_top = H - 4.8*cm
    backdrop_h = 1.3*cm
    c.setFillColor(WHITE)
    c.rect(1.8*cm, name_y_top - 0.35*cm, W - 3.6*cm, backdrop_h, stroke=0, fill=1)

    # Student info (on top of backdrop)
    c.setFillColor(TITLE)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(2*cm, name_y_top, f"{r['First_Name']} {r['Last_Name']}  •  {r['Student_ID']}  •  {dept}")
    c.setFont("Helvetica", 11)
    c.setFillColor(BODY)
    c.drawString(2*cm, name_y_top - 0.7*cm, f"Predicted Final: {pred:.2f}   |   Actual: {float(r['Final_Score']):.2f}")

    # --- Move the radar lower to avoid any chance of overlap ---
    img_x = 2*cm
    img_y = H - 16.0*cm   # lowered from -14.5cm
    c.drawImage(radar_img, img_x, img_y, width=10*cm, height=10*cm, mask='auto')

    # Metrics table (right)
    c.setFont("Helvetica-Bold", 11); c.setFillColor(TITLE)
    c.drawString(12.5*cm, name_y_top - 0.7*cm, "Scores vs Dept Avg")
    y = name_y_top - 1.6*cm
    c.setFont("Helvetica", 10)
    for label, key in [("Midterm","midterm"),("Assignments","assignments"),
                       ("Quizzes","quizzes"),("Projects","projects"),
                       ("Participation","participation")]:
        s = student_metrics[key]; a = dept_avg[key]; d = deltas[key]
        c.setFillColor(MUTED); c.drawString(12.5*cm, y, label)
        c.setFillColor(TITLE); c.drawRightString(19.0*cm, y, f"{s:.1f}  vs  {a:.1f}  ({'+' if d>=0 else ''}{d:.1f})")
        y -= 0.85*cm

    # Suggestions
    c.setFont("Helvetica-Bold", 11); c.setFillColor(TITLE)
    c.drawString(2*cm, H-16.8*cm, "Areas of Improvement & Notes")
    c.setFont("Helvetica", 10); c.setFillColor(BODY)
    y2 = H - 17.8*cm
    for s in suggestions:
        if y2 < 2.5*cm:
            c.showPage()
            c.setFillColor(TITLE); c.setFont("Helvetica-Bold", 12)
            c.drawString(2*cm, H-2.5*cm, "Areas of Improvement & Notes (cont.)")
            c.setFont("Helvetica", 10); c.setFillColor(BODY)
            y2 = H - 3.5*cm
        c.setFillColor(BLUE)
        c.circle(2.05*cm, y2+0.1*cm, 2, stroke=0, fill=1)
        c.setFillColor(BODY)
        c.drawString(2.6*cm, y2, s)
        y2 -= 0.85*cm

    c.showPage(); c.save()
    pdf_buf.seek(0)
    return pdf_buf



# ---------- Download endpoints ----------
@app.route("/report/<student_id>.pdf")
@jwt_required()
def report_student(student_id):
    try:
        _check_access_student(student_id, get_jwt())
        pdf_buf = _student_report_pdf(student_id)
        if pdf_buf is None:
            return jsonify({"error": "Student not found"}), 404
        filename = f"{student_id}_report.pdf"
        return send_file(pdf_buf, mimetype="application/pdf", as_attachment=True, download_name=filename)
    except Forbidden as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        # Avoid 500s without context
        return jsonify({"error": f"Failed to generate report: {e}"}), 500

@app.route("/mentor/<mentor_id>/reports.zip")
@jwt_required()
def reports_zip_for_mentor(mentor_id):
    try:
        claims = get_jwt()
        role = claims.get("role")
        if mentor_id not in MENTOR_BY_ID:
            return jsonify({"error": "Mentor not found"}), 404
        if role == "mentor" and claims.get("mentor_id") != mentor_id:
            raise Forbidden("Mentors can only download their own department")
        dept = MENTOR_BY_ID[mentor_id]["department"]
        df = data[data["Department"] == dept][["Student_ID"]]

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for sid in df["Student_ID"].tolist():
                pdf_buf = _student_report_pdf(sid)
                if pdf_buf is None:
                    continue
                zf.writestr(f"{sid}_report.pdf", pdf_buf.getvalue())
        zip_buf.seek(0)
        return send_file(zip_buf, mimetype="application/zip", as_attachment=True, download_name=f"{mentor_id}_reports.zip")
    except Forbidden as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        return jsonify({"error": f"Failed to build ZIP: {e}"}), 500

@app.route("/mentor/<mentor_id>/report/<student_id>.pdf")
@jwt_required()
def mentor_single_report(mentor_id, student_id):
    try:
        if mentor_id not in MENTOR_BY_ID:
            return jsonify({"error": "Mentor not found"}), 404
        claims = get_jwt()
        role = claims.get("role")
        if role == "mentor" and claims.get("mentor_id") != mentor_id:
            raise Forbidden("Mentors can only download their own department")
        dept = MENTOR_BY_ID[mentor_id]["department"]
        if data.loc[(data["Student_ID"] == student_id) & (data["Department"] == dept)].empty and role != "admin":
            raise Forbidden("Not in your department")
        pdf_buf = _student_report_pdf(student_id)
        if pdf_buf is None:
            return jsonify({"error": "Student not found"}), 404
        return send_file(pdf_buf, mimetype="application/pdf", as_attachment=True, download_name=f"{student_id}_report.pdf")
    except Forbidden as e:
        return jsonify({"error": str(e)}), 403
    except Exception as e:
        return jsonify({"error": f"Failed to generate report: {e}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
