(() => {
  const state = {
    token: localStorage.getItem("enhance_rag_token") || "",
    page: 1,
    pageSize: 20,
    total: 0,
    hasMore: false,
    filters: {
      keyword: "",
      groupScope: "",
      roleId: "",
    },
  };

  const dom = {
    loginView: document.getElementById("login-view"),
    dashboardView: document.getElementById("dashboard-view"),
    loginForm: document.getElementById("login-form"),
    loginError: document.getElementById("login-error"),
    password: document.getElementById("password"),
    logoutBtn: document.getElementById("logout-btn"),
    refreshBtn: document.getElementById("refresh-btn"),
    searchBtn: document.getElementById("search-btn"),
    resetBtn: document.getElementById("reset-btn"),
    keywordInput: document.getElementById("keyword-input"),
    scopeInput: document.getElementById("scope-input"),
    roleInput: document.getElementById("role-input"),
    body: document.getElementById("memory-body"),
    prevBtn: document.getElementById("prev-btn"),
    nextBtn: document.getElementById("next-btn"),
    pageInfo: document.getElementById("page-info"),
    pageSize: document.getElementById("page-size"),
    drawer: document.getElementById("detail-drawer"),
    drawerClose: document.getElementById("drawer-close"),
    drawerContent: document.getElementById("drawer-content"),
    toast: document.getElementById("toast"),
    statTotal: document.getElementById("stat-total"),
    statScopes: document.getElementById("stat-scopes"),
    statGroups: document.getElementById("stat-groups"),
    statRoles: document.getElementById("stat-roles"),
  };

  function init() {
    dom.loginForm.addEventListener("submit", onLogin);
    dom.logoutBtn.addEventListener("click", logout);
    dom.refreshBtn.addEventListener("click", fetchAll);
    dom.searchBtn.addEventListener("click", onSearch);
    dom.resetBtn.addEventListener("click", onReset);
    dom.prevBtn.addEventListener("click", () => {
      if (state.page > 1) {
        state.page -= 1;
        fetchMemories();
      }
    });
    dom.nextBtn.addEventListener("click", () => {
      if (state.hasMore) {
        state.page += 1;
        fetchMemories();
      }
    });
    dom.pageSize.addEventListener("change", () => {
      state.pageSize = Number(dom.pageSize.value || 20);
      state.page = 1;
      fetchMemories();
    });
    dom.drawerClose.addEventListener("click", () => {
      dom.drawer.classList.remove("open");
    });

    if (state.token) {
      switchView("dashboard");
      fetchAll().catch((error) => {
        console.error(error);
        logout();
      });
    } else {
      switchView("login");
    }
  }

  function switchView(target) {
    if (target === "dashboard") {
      dom.loginView.classList.remove("active");
      dom.dashboardView.classList.add("active");
      return;
    }
    dom.dashboardView.classList.remove("active");
    dom.loginView.classList.add("active");
  }

  function showToast(message, isError = false) {
    dom.toast.textContent = message;
    dom.toast.style.background = isError ? "#7f1d1d" : "#111827";
    dom.toast.classList.add("show");
    setTimeout(() => dom.toast.classList.remove("show"), 2200);
  }

  async function api(path, options = {}) {
    const headers = {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    };
    if (!options.skipAuth && state.token) {
      headers.Authorization = `Bearer ${state.token}`;
    }
    const response = await fetch(path, {
      method: options.method || "GET",
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || payload.error || `HTTP ${response.status}`);
    }
    return response.json();
  }

  async function onLogin(event) {
    event.preventDefault();
    dom.loginError.textContent = "";
    try {
      const payload = await api("/api/login", {
        method: "POST",
        body: { password: dom.password.value.trim() },
        skipAuth: true,
      });
      state.token = payload.token;
      localStorage.setItem("enhance_rag_token", state.token);
      dom.password.value = "";
      switchView("dashboard");
      showToast("Login success.");
      await fetchAll();
    } catch (error) {
      dom.loginError.textContent = error.message || "Login failed";
    }
  }

  async function logout() {
    try {
      if (state.token) {
        await api("/api/logout", { method: "POST" });
      }
    } catch (error) {
      console.warn(error);
    } finally {
      state.token = "";
      localStorage.removeItem("enhance_rag_token");
      switchView("login");
    }
  }

  function renderStats(stats) {
    dom.statTotal.textContent = stats.total_memories ?? 0;
    dom.statScopes.textContent = stats.group_scope_count ?? 0;
    dom.statGroups.textContent = stats.group_id_count ?? 0;
    dom.statRoles.textContent = stats.role_count ?? 0;
  }

  function clipText(text, max = 90) {
    const source = String(text || "");
    return source.length > max ? `${source.slice(0, max)}...` : source;
  }

  function renderRows(items) {
    if (!Array.isArray(items) || items.length === 0) {
      dom.body.innerHTML = '<tr><td colspan="6" class="center muted">No records</td></tr>';
      return;
    }

    dom.body.innerHTML = "";
    for (const item of items) {
      const tr = document.createElement("tr");
      const rolePills = (item.related_role_ids || [])
        .map((role) => `<span class="pill mono">${escapeHtml(role)}</span>`)
        .join("");
      tr.innerHTML = `
        <td class="mono">${item.memory_id}</td>
        <td>${escapeHtml(clipText(item.content, 120))}</td>
        <td class="mono">${escapeHtml(item.group_scope || "-")}</td>
        <td>${rolePills || "<span class='muted'>-</span>"}</td>
        <td class="mono">${escapeHtml(item.memory_time_iso || "-")}</td>
        <td>
          <div class="action-group">
            <button class="ghost" data-action="detail" data-id="${item.memory_id}">Detail</button>
            <button class="danger" data-action="delete" data-id="${item.memory_id}">Delete</button>
          </div>
        </td>
      `;
      dom.body.appendChild(tr);
    }

    dom.body.querySelectorAll("button[data-action='detail']").forEach((btn) => {
      btn.addEventListener("click", () => openDetail(Number(btn.dataset.id)));
    });
    dom.body.querySelectorAll("button[data-action='delete']").forEach((btn) => {
      btn.addEventListener("click", () => deleteMemory(Number(btn.dataset.id)));
    });
  }

  async function fetchStats() {
    const payload = await api("/api/stats");
    renderStats(payload.data || {});
  }

  async function fetchMemories() {
    const params = new URLSearchParams();
    params.set("page", String(state.page));
    params.set("page_size", String(state.pageSize));
    if (state.filters.keyword) params.set("keyword", state.filters.keyword);
    if (state.filters.groupScope) params.set("group_scope", state.filters.groupScope);
    if (state.filters.roleId) params.set("role_id", state.filters.roleId);

    const payload = await api(`/api/memories?${params.toString()}`);
    const data = payload.data || {};
    state.total = Number(data.total || 0);
    state.hasMore = Boolean(data.has_more);
    renderRows(data.items || []);
    const totalPages = Math.max(1, Math.ceil(state.total / state.pageSize));
    dom.pageInfo.textContent = `Page ${state.page}/${totalPages} · Total ${state.total}`;
    dom.prevBtn.disabled = state.page <= 1;
    dom.nextBtn.disabled = !state.hasMore;
  }

  async function fetchAll() {
    try {
      await Promise.all([fetchStats(), fetchMemories()]);
    } catch (error) {
      showToast(error.message || "Load failed", true);
      if (String(error.message || "").toLowerCase().includes("token")) {
        logout();
      }
    }
  }

  function onSearch() {
    state.filters.keyword = dom.keywordInput.value.trim();
    state.filters.groupScope = dom.scopeInput.value.trim();
    state.filters.roleId = dom.roleInput.value.trim();
    state.page = 1;
    fetchMemories();
  }

  function onReset() {
    dom.keywordInput.value = "";
    dom.scopeInput.value = "";
    dom.roleInput.value = "";
    state.filters.keyword = "";
    state.filters.groupScope = "";
    state.filters.roleId = "";
    state.page = 1;
    fetchMemories();
  }

  async function openDetail(memoryId) {
    try {
      const payload = await api(`/api/memories/${memoryId}`);
      dom.drawerContent.textContent = JSON.stringify(payload.data || {}, null, 2);
      dom.drawer.classList.add("open");
    } catch (error) {
      showToast(error.message || "Detail load failed", true);
    }
  }

  async function deleteMemory(memoryId) {
    const confirmed = window.confirm(`Delete memory ${memoryId}?`);
    if (!confirmed) return;
    try {
      await api(`/api/memories/${memoryId}`, { method: "DELETE" });
      showToast(`Deleted memory ${memoryId}`);
      await fetchAll();
    } catch (error) {
      showToast(error.message || "Delete failed", true);
    }
  }

  function escapeHtml(text) {
    return String(text || "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  init();
})();
