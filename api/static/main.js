// FaceShield AI — Main Application Script
// Drives the Portal (Login/Signup + Face CAPTCHA) and Admin HUD tabs.

document.addEventListener("DOMContentLoaded", () => {

  // ── Tab elements ──────────────────────────────────────────────────
  const btnTabPortal = document.getElementById("btn-tab-portal");
  const btnTabAdmin  = document.getElementById("btn-tab-admin");
  const tabPortal    = document.getElementById("tab-portal");
  const tabAdmin     = document.getElementById("tab-admin");

  // ── Header status ─────────────────────────────────────────────────
  const statusPill = document.getElementById("status-pill");
  const statusDot  = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const fpsChip    = document.getElementById("fps-chip");

  // ── Portal views ──────────────────────────────────────────────────
  const portalGate    = document.getElementById("portal-gate");
  const portalVerify  = document.getElementById("portal-verify");
  const portalSuccess = document.getElementById("portal-success");

  const viewLogin  = document.getElementById("view-login");
  const viewSignup = document.getElementById("view-signup");
  const heroLogin  = document.getElementById("hero-login");
  const heroSignup = document.getElementById("hero-signup");

  // Login form
  const formLogin     = document.getElementById("form-login");
  const loginEmail    = document.getElementById("login-email");
  const loginPw       = document.getElementById("login-password");
  const loginMsg      = document.getElementById("login-msg");
  const loginBtn      = document.getElementById("login-submit-btn");

  // Signup form
  const formSignup    = document.getElementById("form-signup");
  const signupName    = document.getElementById("signup-name");
  const signupEmail   = document.getElementById("signup-email");
  const signupPw      = document.getElementById("signup-password");
  const signupConfirm = document.getElementById("signup-confirm");
  const signupFile    = document.getElementById("signup-face-file");
  const signupDzText  = document.getElementById("signup-dz-text");
  const signupDzone   = document.getElementById("signup-dropzone");
  const signupMsg     = document.getElementById("signup-msg");
  const signupBtn     = document.getElementById("signup-submit-btn");

  // Verify page
  const verifyFeed   = document.getElementById("verify-feed");
  const snapCanvas   = document.getElementById("snap-canvas");
  const verifyMsg    = document.getElementById("verify-msg");
  const ringPath     = document.getElementById("ring-path");
  const ringPct      = document.getElementById("ring-pct");
  const ringStatus   = document.getElementById("ring-status");
  const btnVerifyStr = document.getElementById("btn-verify-stream");
  const btnVerifySnap= document.getElementById("btn-verify-snap");
  const btnBackLogin = document.getElementById("btn-back-to-login");

  // Success page
  const successUsername = document.getElementById("success-username");
  const btnLogout       = document.getElementById("btn-logout");

  // Admin
  const adminFeed    = document.getElementById("admin-feed");
  const feedHud      = document.getElementById("feed-hud");
  const dIcon        = document.getElementById("d-icon");
  const dLabel       = document.getElementById("d-label");
  const dUser        = document.getElementById("d-user");
  const barLive      = document.getElementById("bar-live");
  const barIden      = document.getElementById("bar-iden");
  const barComb      = document.getElementById("bar-comb");
  const valLive      = document.getElementById("val-live");
  const valIden      = document.getElementById("val-iden");
  const valComb      = document.getElementById("val-comb");
  const reasonText   = document.getElementById("reason-text");
  const formEnroll   = document.getElementById("form-enroll");
  const enrollUsr    = document.getElementById("enroll-username");
  const enrollFile   = document.getElementById("enroll-file");
  const enrollDz     = document.getElementById("enroll-dropzone");
  const enrollDzText = document.getElementById("enroll-dz-text");
  const enrollMsg    = document.getElementById("enroll-msg");
  const auditTbody   = document.getElementById("audit-tbody");
  const btnRefAudit  = document.getElementById("btn-refresh-audit");
  const btnClearUsr  = document.getElementById("btn-clear-users");
  const btnClearAud  = document.getElementById("btn-clear-audit");

  // ── State ─────────────────────────────────────────────────────────
  let loggedInEmail    = "";
  let loggedInUsername = "";
  let statusInterval   = null;
  let auditInterval    = null;
  let verifyPolling    = null;

  // ══════════════════════════════════════════════════════════════════
  // TAB SWITCHING
  // ══════════════════════════════════════════════════════════════════
  function showTab(tab) {
    if (tab === "portal") {
      btnTabPortal.classList.add("active");
      btnTabAdmin.classList.remove("active");
      tabPortal.classList.remove("hidden");
      tabAdmin.classList.add("hidden");
      // Release admin feed when not on admin tab
      adminFeed.src = "";
    } else {
      btnTabAdmin.classList.add("active");
      btnTabPortal.classList.remove("active");
      tabAdmin.classList.remove("hidden");
      tabPortal.classList.add("hidden");
      // Activate admin camera feed
      adminFeed.src = "/video_feed";
      fetchAuditLog();
    }
  }

  btnTabPortal.addEventListener("click", () => showTab("portal"));
  btnTabAdmin.addEventListener("click",  () => showTab("admin"));

  // ══════════════════════════════════════════════════════════════════
  // PORTAL NAVIGATION
  // ══════════════════════════════════════════════════════════════════
  function showGate(which) {
    // which = "login" | "signup"
    portalGate.classList.remove("hidden");
    portalVerify.classList.add("hidden");
    portalSuccess.classList.add("hidden");

    if (which === "login") {
      viewLogin.classList.remove("hidden");
      viewSignup.classList.add("hidden");
      heroLogin.classList.remove("hidden");
      heroSignup.classList.add("hidden");
    } else {
      viewSignup.classList.remove("hidden");
      viewLogin.classList.add("hidden");
      heroSignup.classList.remove("hidden");
      heroLogin.classList.add("hidden");
    }
  }

  function showVerify() {
    portalGate.classList.add("hidden");
    portalVerify.classList.remove("hidden");
    portalSuccess.classList.add("hidden");
    // Start the camera stream for verification
    verifyFeed.src = "/video_feed";
    startVerifyPolling();
  }

  function showSuccess(username, lr) {
    portalGate.classList.add("hidden");
    portalVerify.classList.add("hidden");
    portalSuccess.classList.remove("hidden");
    successUsername.textContent = `@${username}`;

    // Populate biometric score badges if available
    const liveEl  = document.getElementById("success-live-score");
    const idEl    = document.getElementById("success-id-score");
    const combEl  = document.getElementById("success-comb-score");
    if (liveEl && lr) {
      liveEl.textContent  = lr.liveness_score  !== undefined ? (lr.liveness_score  * 100).toFixed(0) + "% Liveness"  : "—";
      idEl.textContent    = lr.identity_score  !== undefined ? (lr.identity_score  * 100).toFixed(0) + "% Identity"  : "—";
      combEl.textContent  = lr.combined_score  !== undefined ? (lr.combined_score  * 100).toFixed(0) + "% Combined"  : "—";
    }

    // Trigger confetti burst
    launchConfetti();

    // Release verify camera stream
    verifyFeed.src = "";
    stopVerifyPolling();
  }

  // Lightweight canvas confetti
  function launchConfetti() {
    const canvas = document.getElementById("confetti-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
    canvas.style.display = "block";
    const pieces = Array.from({ length: 120 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height * -0.5,
      r: 5 + Math.random() * 8,
      d: 1 + Math.random() * 2.5,
      color: ["#00e676","#7c4dff","#00e5ff","#ffab40","#ff4081"][Math.floor(Math.random()*5)],
      tilt: Math.random() * 30 - 15,
    }));
    let frame = 0;
    function draw() {
      if (frame++ > 160) { canvas.style.display = "none"; return; }
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      pieces.forEach(p => {
        p.y += p.d;
        p.tilt += 0.12;
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, p.r, p.r * 0.5, p.tilt, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.fill();
      });
      requestAnimationFrame(draw);
    }
    draw();
  }

  document.getElementById("go-signup").addEventListener("click", () => showGate("signup"));
  document.getElementById("go-login").addEventListener("click",  () => showGate("login"));

  btnBackLogin.addEventListener("click", () => {
    verifyFeed.src = "";
    stopVerifyPolling();
    showGate("login");
  });

  btnLogout.addEventListener("click", () => {
    loggedInEmail    = "";
    loggedInUsername = "";
    showGate("login");
  });

  // ══════════════════════════════════════════════════════════════════
  // PASSWORD EYE TOGGLES
  // ══════════════════════════════════════════════════════════════════
  function makeToggle(toggleId, inputEl) {
    const btn = document.getElementById(toggleId);
    if (!btn) return;
    btn.addEventListener("click", () => {
      inputEl.type = inputEl.type === "password" ? "text" : "password";
      btn.textContent = inputEl.type === "password" ? "👁️" : "🙈";
    });
  }
  makeToggle("toggle-login-pw",   loginPw);
  makeToggle("toggle-signup-pw1", signupPw);
  makeToggle("toggle-signup-pw2", signupConfirm);

  // ══════════════════════════════════════════════════════════════════
  // DROPZONE HELPERS
  // ══════════════════════════════════════════════════════════════════
  function wireDropzone(dzone, fileInput, dzTextEl) {
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) {
        const name = fileInput.files[0].name;
        dzTextEl.textContent = `✅ ${name}`;
        dzone.classList.add("done");
      }
    });
    dzone.addEventListener("dragover", e => { e.preventDefault(); dzone.classList.add("over"); });
    dzone.addEventListener("dragleave", () => dzone.classList.remove("over"));
    dzone.addEventListener("drop", e => {
      e.preventDefault();
      dzone.classList.remove("over");
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event("change"));
      }
    });
  }
  wireDropzone(signupDzone, signupFile, signupDzText);
  wireDropzone(enrollDz, enrollFile, enrollDzText);

  // ══════════════════════════════════════════════════════════════════
  // PORTAL: SIGNUP
  // ══════════════════════════════════════════════════════════════════
  formSignup.addEventListener("submit", async e => {
    e.preventDefault();
    clearMsg(signupMsg);

    if (signupPw.value !== signupConfirm.value) {
      showMsg(signupMsg, "Passwords do not match.", "error");
      return;
    }
    if (!signupFile.files.length) {
      showMsg(signupMsg, "Please upload a face photo.", "error");
      return;
    }

    const fd = new FormData();
    fd.append("username", signupName.value.trim());
    fd.append("email",    signupEmail.value.trim());
    fd.append("password", signupPw.value);
    fd.append("image",    signupFile.files[0]);

    setLoading(signupBtn, true);
    try {
      const res = await fetch("/api/portal/signup", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Signup failed.");
      showMsg(signupMsg, `✅ ${data.message} Redirecting to login…`, "success");
      setTimeout(() => showGate("login"), 2200);
    } catch (err) {
      showMsg(signupMsg, `❌ ${err.message}`, "error");
    } finally {
      setLoading(signupBtn, false);
    }
  });

  // ══════════════════════════════════════════════════════════════════
  // PORTAL: LOGIN (Step 1 — credentials)
  // ══════════════════════════════════════════════════════════════════
  formLogin.addEventListener("submit", async e => {
    e.preventDefault();
    clearMsg(loginMsg);

    const fd = new FormData();
    fd.append("email",    loginEmail.value.trim());
    fd.append("password", loginPw.value);

    setLoading(loginBtn, true);
    try {
      const res  = await fetch("/api/portal/login/password", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Login failed.");

      loggedInEmail    = loginEmail.value.trim();
      loggedInUsername = data.username;

      showMsg(loginMsg, `✅ Credentials verified! Opening face scanner…`, "success");
      setTimeout(() => showVerify(), 1200);
    } catch (err) {
      showMsg(loginMsg, `❌ ${err.message}`, "error");
    } finally {
      setLoading(loginBtn, false);
    }
  });

  // ══════════════════════════════════════════════════════════════════
  // PORTAL: FACE VERIFICATION (Step 2)
  // ══════════════════════════════════════════════════════════════════

  // Ring progress update
  function setRing(pct, label, color) {
    ringPath.setAttribute("stroke-dasharray", `${pct}, 100`);
    ringPct.textContent = `${Math.round(pct)}%`;
    if (label) ringStatus.textContent = label;
    if (color) ringPath.style.stroke = color;
  }

  // Auto-verify: poll pipeline status and auto-redirect on grant
  let _autoGranted = false;
  function startVerifyPolling() {
    _autoGranted = false;
    setRing(0, "📡 Searching for your face…", "#00e5ff");
    showMsg(verifyMsg, "Look at the camera — face detection is running automatically.", "info");

    verifyPolling = setInterval(async () => {
      if (_autoGranted) return;   // already redirecting
      try {
        const res  = await fetch("/api/status");
        const data = await res.json();

        const bufPct = data.seq_len > 0
          ? Math.round((data.buffer_fill / data.seq_len) * 100)
          : 0;

        const lr = data.last_result;   // last pipeline decision

        if (!data.face_detected) {
          setRing(Math.min(bufPct, 20), "📡 Searching for your face…", "#00e5ff");
          clearMsg(verifyMsg);
          showMsg(verifyMsg, "Look directly at the camera.", "info");

        } else if (lr && lr.granted && lr.username && lr.username.toLowerCase() === loggedInUsername.toLowerCase()) {
          // ─── ACCESS GRANTED ────────────────────────────────────────
          _autoGranted = true;
          stopVerifyPolling();

          setRing(100, "✅ Identity Confirmed!", "#00e676");
          clearMsg(verifyMsg);
          showMsg(verifyMsg, "✅ Face matched! Redirecting…", "success");

          // brief pause so user sees 100% ring, then redirect
          setTimeout(() => showSuccess(loggedInUsername, lr), 1400);

        } else if (data.face_detected && !(lr && lr.granted)) {
          // Face seen but decision pending or denied — animate ring fill
          const displayPct = Math.max(bufPct, 30);
          if (lr && lr.liveness_score !== undefined && lr.liveness_score < 0.4) {
            setRing(displayPct, "⚠️ Liveness check…", "#ffab40");
            showMsg(verifyMsg, "Hold still and look at the camera.", "warn");
          } else {
            setRing(displayPct, "🔍 Checking identity…", "#7c4dff");
            clearMsg(verifyMsg);
            showMsg(verifyMsg, "Face detected — matching identity…", "info");
          }
        } else {
          setRing(bufPct, "🔍 Analysing…", "#00e5ff");
        }
      } catch (_) {}
    }, 600);
  }

  function stopVerifyPolling() {
    if (verifyPolling) { clearInterval(verifyPolling); verifyPolling = null; }
  }

  // Live-stream verify button
  btnVerifyStr.addEventListener("click", async () => {
    clearMsg(verifyMsg);
    setLoading(btnVerifyStr, true);
    try {
      const fd = new FormData();
      fd.append("email", loggedInEmail);
      const res  = await fetch("/api/portal/login/face-stream", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Face verification failed.");
      setRing(100, "✅ Verified!");
      showMsg(verifyMsg, `✅ Face matched! Logging in…`, "success");
      setTimeout(() => showSuccess(loggedInUsername), 1200);
    } catch (err) {
      showMsg(verifyMsg, `❌ ${err.message}`, "error");
      setRing(0, "Retry…");
    } finally {
      setLoading(btnVerifyStr, false);
    }
  });

  // Snapshot verify button
  btnVerifySnap.addEventListener("click", async () => {
    clearMsg(verifyMsg);
    const ctx = snapCanvas.getContext("2d");
    ctx.drawImage(verifyFeed, 0, 0, snapCanvas.width, snapCanvas.height);
    snapCanvas.toBlob(async blob => {
      if (!blob) { showMsg(verifyMsg, "❌ Could not capture frame.", "error"); return; }
      const fd = new FormData();
      fd.append("email", loggedInEmail);
      fd.append("image", blob, "snapshot.jpg");
      setLoading(btnVerifySnap, true);
      try {
        const res  = await fetch("/api/portal/login/face-upload", { method: "POST", body: fd });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Face verification failed.");
        setRing(100, "✅ Verified!");
        showMsg(verifyMsg, `✅ Face matched (${(data.similarity*100).toFixed(1)}%)! Logging in…`, "success");
        setTimeout(() => showSuccess(loggedInUsername), 1200);
      } catch (err) {
        showMsg(verifyMsg, `❌ ${err.message}`, "error");
      } finally {
        setLoading(btnVerifySnap, false);
      }
    }, "image/jpeg", 0.85);
  });

  // ══════════════════════════════════════════════════════════════════
  // ADMIN: ENROLL FORM
  // ══════════════════════════════════════════════════════════════════
  formEnroll.addEventListener("submit", async e => {
    e.preventDefault();
    clearMsg(enrollMsg);
    if (!enrollFile.files.length) { showMsg(enrollMsg, "Please select an image.", "error"); return; }

    const fd = new FormData();
    fd.append("username", enrollUsr.value.trim());
    fd.append("image",    enrollFile.files[0]);

    try {
      const res  = await fetch("/api/enroll", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Enroll failed.");
      showMsg(enrollMsg, `✅ ${data.message}`, "success");
      formEnroll.reset();
      enrollDzText.textContent = "Click or drag image here";
      enrollDz.classList.remove("done");
    } catch (err) {
      showMsg(enrollMsg, `❌ ${err.message}`, "error");
    }
  });

  // Clear users / audit
  btnClearUsr.addEventListener("click", async () => {
    if (!confirm("Delete ALL enrolled users?")) return;
    await fetch("/api/users", { method: "DELETE" });
    showMsg(enrollMsg, "✅ All users cleared.", "success");
  });
  btnClearAud.addEventListener("click", async () => {
    await fetch("/api/audit", { method: "DELETE" });
    fetchAuditLog();
  });

  // ══════════════════════════════════════════════════════════════════
  // ADMIN: AUDIT LOG
  // ══════════════════════════════════════════════════════════════════
  async function fetchAuditLog() {
    try {
      const res  = await fetch("/api/audit?limit=15");
      const data = await res.json();
      if (!Array.isArray(data) || data.length === 0) {
        auditTbody.innerHTML = `<tr><td colspan="4" class="empty-row">No records yet.</td></tr>`;
        return;
      }
      auditTbody.innerHTML = data.map(r => {
        const time = r.timestamp ? r.timestamp.slice(11, 19) : "—";
        const badge = r.decision === "GRANTED"
          ? `<span class="badge-granted">GRANTED</span>`
          : `<span class="badge-denied">DENIED</span>`;
        return `<tr>
          <td class="mono">${time}</td>
          <td>${r.username_claimed}</td>
          <td>${badge}</td>
          <td class="mono">${(r.liveness_score*100).toFixed(0)}/${(r.identity_score*100).toFixed(0)}</td>
        </tr>`;
      }).join("");
    } catch (_) {}
  }

  btnRefAudit.addEventListener("click", fetchAuditLog);

  // ══════════════════════════════════════════════════════════════════
  // STATUS POLLING (header pill + admin decision panel)
  // ══════════════════════════════════════════════════════════════════
  async function pollStatus() {
    try {
      const res  = await fetch("/api/status");
      const data = await res.json();

      // Header
      fpsChip.textContent = `${data.fps} FPS`;
      statusDot.className = "status-dot " + (data.running ? "on" : "off");
      statusText.textContent = data.running
        ? (data.face_detected ? "Face Detected" : "Pipeline Active")
        : "Pipeline Offline";

      // Admin decision panel (only update if admin tab is active)
      if (!tabAdmin.classList.contains("hidden") && data.last_result) {
        const r = data.last_result;
        feedHud.textContent = r.granted
          ? `✅ GRANTED — ${r.username}`
          : `❌ DENIED — ${r.reason || ""}`;
        feedHud.style.color = r.granted ? "var(--green)" : "var(--red)";

        dIcon.textContent  = r.granted ? "✅" : "❌";
        dLabel.textContent = r.decision;
        dLabel.style.color = r.granted ? "var(--green)" : "var(--red)";
        dUser.textContent  = r.username;

        const lv = Math.min(100, r.liveness_score * 100);
        const iv = Math.min(100, r.identity_score * 100);
        const cv = Math.min(100, r.combined_score * 100);
        barLive.style.width = `${lv}%`;
        barIden.style.width = `${iv}%`;
        barComb.style.width = `${cv}%`;
        valLive.textContent = `${lv.toFixed(0)}%`;
        valIden.textContent = `${iv.toFixed(0)}%`;
        valComb.textContent = `${cv.toFixed(0)}%`;
        reasonText.textContent = r.reason || "";
      } else if (!tabAdmin.classList.contains("hidden") && !data.last_result) {
        feedHud.textContent = data.face_detected ? "Scanning…" : "NO TARGET DETECTED";
        feedHud.style.color = "";
      }

    } catch (_) {
      statusDot.className = "status-dot off";
      statusText.textContent = "Connection Error";
    }
  }

  // Start polling immediately
  pollStatus();
  statusInterval = setInterval(pollStatus, 1500);

  // ══════════════════════════════════════════════════════════════════
  // HELPERS
  // ══════════════════════════════════════════════════════════════════
  function showMsg(el, text, type) {
    el.textContent = text;
    el.className = "form-msg " + type;
  }
  function clearMsg(el) {
    el.textContent = "";
    el.className = "form-msg";
  }
  function setLoading(btn, loading) {
    btn.disabled = loading;
    btn.classList.toggle("loading", loading);
  }

});
