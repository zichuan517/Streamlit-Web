# app.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="随机过程：布朗运动及其可视化模拟",
    layout="wide"
)


# Utilities
def _spd_corr_matrix(dim: int, rho: float) -> np.ndarray:
    rho = float(np.clip(rho, -0.99, 0.99))
    lower = -1.0 / (dim - 1) + 1e-6
    if rho <= lower:
        st.warning(f"ρ={rho:.2f} 过小，已自动调整为 {lower:.2f}")
        rho = lower
    C = np.full((dim, dim), rho, dtype=float)
    np.fill_diagonal(C, 1.0)
    return C


def _plot_3d_swarm(paths: np.ndarray, t_idx: int, n_traj: int, title: str) -> go.Figure:
    T, N, _ = paths.shape
    t_idx = int(np.clip(t_idx, 0, T - 1))

    pts = paths[t_idx]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=3, opacity=0.75),
            name=f"粒子位置（t={t_idx}）"
        )
    )

    n_traj = int(np.clip(n_traj, 0, min(N, 50)))
    if n_traj > 0 and t_idx >= 1:
        pick = np.linspace(0, N - 1, n_traj, dtype=int)
        for k, pid in enumerate(pick):
            tr = paths[:t_idx + 1, pid, :]
            fig.add_trace(
                go.Scatter3d(
                    x=tr[:, 0], y=tr[:, 1], z=tr[:, 2],
                    mode="lines",
                    name=f"轨迹 {pid}",
                    showlegend=(k < 5),
                    line=dict(width=3),
                    opacity=0.9
                )
            )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data"
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig


def _line_plot(series: np.ndarray, title: str, yname: str) -> go.Figure:
    x = np.arange(len(series))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=series, mode="lines", name=yname))
    fig.update_layout(
        title=title,
        xaxis_title="step",
        yaxis_title=yname,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def _plot_single_trajectory(paths: np.ndarray, pid: int, title: str) -> go.Figure:
    T, N, _ = paths.shape
    pid = int(np.clip(pid, 0, N - 1))
    tr = paths[:, pid, :]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=tr[:, 0],
            y=tr[:, 1],
            z=tr[:, 2],
            mode="lines+markers",
            marker=dict(
                size=2,
                color=np.arange(len(tr)),
                colorscale="Viridis"
            ),
            line=dict(width=4),
            name=f"粒子 {pid}"
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[tr[0, 0]], y=[tr[0, 1]], z=[tr[0, 2]],
            mode="markers",
            marker=dict(size=6, color="green"),
            name="t=0"
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[tr[-1, 0]], y=[tr[-1, 1]], z=[tr[-1, 2]],
            mode="markers",
            marker=dict(size=6, color="red"),
            name="t=T"
        )
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data"
        ),
        height=500,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig


# Simulators

@st.cache_data(show_spinner=False, ttl=600)
def sim_standard_bm_3d(n_particles, steps, dt, sigma, seed):
    rng = np.random.default_rng(seed)
    dt_sqrt = np.sqrt(dt)
    dW = rng.normal(0.0, 1.0, size=(steps, n_particles, 3))
    X = np.zeros((steps + 1, n_particles, 3))
    X[1:] = sigma * dt_sqrt * np.cumsum(dW, axis=0)
    return X


@st.cache_data(show_spinner=False, ttl=600)
def sim_drift_bm_3d(n_particles, steps, dt, mu, sigma, seed):
    rng = np.random.default_rng(seed)
    dt_sqrt = np.sqrt(dt)
    dW = rng.normal(0.0, 1.0, size=(steps, n_particles, 3))
    X = np.zeros((steps + 1, n_particles, 3))
    X[1:] = np.cumsum(mu * dt + sigma * dt_sqrt * dW, axis=0)
    return X


@st.cache_data(show_spinner=False, ttl=600)
def sim_correlated_bm_3d(n_particles, steps, dt, sigma, rho, seed):
    rng = np.random.default_rng(seed)
    dt_sqrt = np.sqrt(dt)
    C = _spd_corr_matrix(3, rho)
    L = np.linalg.cholesky(C)
    Z = rng.normal(size=(steps, n_particles, 3))
    dB = Z @ L.T
    X = np.zeros((steps + 1, n_particles, 3))
    X[1:] = sigma * dt_sqrt * np.cumsum(dB, axis=0)
    return X


@st.cache_data(show_spinner=False, ttl=600)
def sim_gbm_3d(n_particles, steps, dt, mu, sigma, rho, x0, seed):
    rng = np.random.default_rng(seed)
    dt_sqrt = np.sqrt(dt)
    C = _spd_corr_matrix(3, rho)
    L = np.linalg.cholesky(C)
    Z = rng.normal(size=(steps, n_particles, 3))
    dB = Z @ L.T
    logX = np.zeros((steps + 1, n_particles, 3))
    logX[0] = np.log(max(x0, 1e-9))
    logX[1:] = np.cumsum(
        (mu - 0.5 * sigma**2) * dt + sigma * dt_sqrt * dB,
        axis=0
    ) + logX[0]
    return np.exp(logX)

# Sidebar

st.title("随机过程：布朗运动及其可视化模拟")

with st.sidebar:
    st.header("全局参数")
    steps = st.slider("步数", 50, 600, 240, 10)
    dt = st.slider("时间步长", 0.001, 0.05, 0.01, 0.001)
    n_particles = st.slider("粒子数", 50, 2000, 500, 50)
    show_traj = st.slider("轨迹条数", 0, 30, 8, 1)
    seed = st.number_input("随机种子", 0, 10_000_000, 7)
    pid_single = st.slider("单粒子编号", 0, n_particles - 1, 0)

    st.divider()
    st.subheader("模型参数")
    sigma = st.slider("σ", 0.1, 3.0, 1.0, 0.05)
    mu = st.slider("μ", -2.0, 2.0, 0.3, 0.05)
    rho = st.slider("ρ", -0.9, 0.9, 0.5, 0.05)
    x0 = st.slider("X(0)", 0.1, 10.0, 1.0, 0.1)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["标准布朗运动", "漂移布朗运动", "相关布朗运动", "几何布朗运动", "理论验证"]
)

# Tab 1
with tab1:
    st.subheader("标准布朗运动（维纳过程）")
    st.latex(r"B(0)=0,\quad B(t)-B(s)\sim\mathcal N(0,t-s)")

    X = sim_standard_bm_3d(n_particles, steps, dt, sigma, seed)

    t_idx = st.slider("时间索引 t", 0, steps, steps // 2, key="t1")

    st.plotly_chart(
        _plot_3d_swarm(X, t_idx, show_traj, "标准布朗运动：三维粒子群"),
        width="stretch"
    )

    st.markdown("### 单粒子轨迹（1D / 2D / 3D）")

    tr = X[:, pid_single, :]
    t = np.arange(steps + 1) * dt

    fig1d = go.Figure()
    fig1d.add_trace(go.Scatter(x=t, y=tr[:, 0], mode="lines"))
    fig1d.update_layout(
        title="1D 标准布朗运动轨迹",
        xaxis_title="t",
        yaxis_title="x(t)"
    )

    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=tr[:, 0], y=tr[:, 1], mode="lines"))
    fig2d.update_layout(
        title="2D 标准布朗运动轨迹",
        xaxis_title="x",
        yaxis_title="y",
        xaxis=dict(scaleanchor="y", scaleratio=1)
    )

    fig3d = _plot_single_trajectory(
        X,
        pid_single,
        "3D 标准布朗运动轨迹"
    )

    st.plotly_chart(fig1d, width="stretch")
    st.plotly_chart(fig2d, width="stretch")
    st.plotly_chart(fig3d, width="stretch")

# Tab 2
with tab2:
    st.subheader("漂移布朗运动")
    st.latex(r"X(t)=\mu t+\sigma B(t)")
    st.markdown(
        r"- **直观解释**：随机抖动上叠加确定性线性趋势。" "\n"
        r"- $\mu>0$ 向上漂移，$\mu<0$ 向下漂移。"
    )

    X = sim_drift_bm_3d(n_particles, steps, dt, mu, sigma, seed + 1)
    t_idx = st.slider("时间索引 t", 0, steps, steps // 2, key="t2")
    st.plotly_chart(
        _plot_3d_swarm(X, t_idx, show_traj, "漂移布朗运动：三维粒子群"),
        width="stretch"
    )
    st.markdown("#### 单个粒子的三维完整轨迹")
    st.plotly_chart(
        _plot_single_trajectory(
            X,
            pid_single,
            "漂移布朗运动：单粒子三维轨迹"
        ),
        width="stretch"
    )

# Tab 3
with tab3:
    st.subheader("相关布朗运动")
    st.latex(r"\mathrm{Cov}(B_i,B_j)=\rho_{ij}t")
    st.markdown(
        r"- **直观解释**：不同方向的随机扰动不再独立，而是具有相关性。" "\n"
        r"- $\rho>0$ 同涨同跌，$\rho<0$ 相互牵制。"
    )

    X = sim_correlated_bm_3d(n_particles, steps, dt, sigma, rho, seed + 2)
    t_idx = st.slider("时间索引 t", 0, steps, steps // 2, key="t3")
    st.plotly_chart(
        _plot_3d_swarm(X, t_idx, show_traj, f"相关布朗运动"),
        width="stretch"
    )
    st.markdown("#### 单个粒子的三维完整轨迹")
    st.plotly_chart(
        _plot_single_trajectory(
            X,
            pid_single,
            f"相关布朗运动：单粒子三维轨迹"
        ),
        width="stretch"
    )

# Tab 4
with tab4:
    st.subheader("几何布朗运动")
    st.latex(r"\mathrm dX=\mu X\,\mathrm dt+\sigma X\,\mathrm dB")
    st.markdown(
        r"- **直观解释**：随机扰动按比例作用，状态始终保持正值。" "\n"
        r"- 常用于描述价格、规模等指数型随机演化。"
    )

    X = sim_gbm_3d(n_particles, steps, dt, mu, sigma, rho, x0, seed + 3)
    t_idx = st.slider("时间索引 t", 0, steps, steps // 2, key="t4")
    st.plotly_chart(
        _plot_3d_swarm(X, t_idx, show_traj, "几何布朗运动：三维粒子群"),
        width="stretch"
    )
    st.markdown("#### 单个粒子的三维完整轨迹")
    st.plotly_chart(
        _plot_single_trajectory(
            X,
            pid_single,
            "几何布朗运动：单粒子三维轨迹"
        ),
        width="stretch"
    )

# Tab 5
with tab5:
    st.subheader("理论结果 vs 数值模拟验证")

    X = sim_standard_bm_3d(n_particles, steps, dt, sigma, seed)
    t = np.arange(steps + 1) * dt

    st.markdown("### 1. 标准布朗运动：均值与方差")
    st.latex(r"\mathbb{E}[X(t)] = 0,\qquad \mathrm{Var}[X(t)] = \sigma^2 t")

    mean_sim = X[:, :, 0].mean(axis=1)
    var_sim = X[:, :, 0].var(axis=1)

    mean_theory = np.zeros_like(t)
    var_theory = sigma**2 * t

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=t, y=mean_sim, mode="lines", name="模拟均值"))
    fig1.add_trace(go.Scatter(x=t, y=mean_theory, mode="lines",
                              name="理论均值",
                              line=dict(dash="dash")))
    fig1.update_layout(title="标准布朗运动：均值", xaxis_title="t", yaxis_title="均值")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=var_sim, mode="lines", name="模拟方差"))
    fig2.add_trace(go.Scatter(x=t, y=var_theory, mode="lines",
                              name="理论方差",
                              line=dict(dash="dash")))
    fig2.update_layout(title="标准布朗运动：方差", xaxis_title="t", yaxis_title="方差")

    st.plotly_chart(fig1, width="stretch")
    st.plotly_chart(fig2, width="stretch")

    st.markdown("### 2. 标准布朗运动：均方位移")
    st.latex(r"\langle r^2(t) \rangle = d\,\sigma^2 t \quad (d=1,2,3)")

    r2_1d = (X[:, :, 0] ** 2).mean(axis=1)
    r2_2d = (X[:, :, 0] ** 2 + X[:, :, 1] ** 2).mean(axis=1)
    r2_3d = (X ** 2).sum(axis=2).mean(axis=1)

    r2_1d_th = sigma**2 * t
    r2_2d_th = 2 * sigma**2 * t
    r2_3d_th = 3 * sigma**2 * t

    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Scatter(x=t, y=r2_1d, mode="lines", name="1D 模拟"))
    fig_r2.add_trace(go.Scatter(x=t, y=r2_1d_th, mode="lines",
                                name="1D 理论",
                                line=dict(dash="dash")))

    fig_r2.add_trace(go.Scatter(x=t, y=r2_2d, mode="lines", name="2D 模拟"))
    fig_r2.add_trace(go.Scatter(x=t, y=r2_2d_th, mode="lines",
                                name="2D 理论",
                                line=dict(dash="dash")))

    fig_r2.add_trace(go.Scatter(x=t, y=r2_3d, mode="lines", name="3D 模拟"))
    fig_r2.add_trace(go.Scatter(x=t, y=r2_3d_th, mode="lines",
                                name="3D 理论",
                                line=dict(dash="dash")))

    fig_r2.update_layout(
        title="标准布朗运动：均方位移扩散定律",
        xaxis_title="t",
        yaxis_title="⟨r²(t)⟩"
    )

    st.plotly_chart(fig_r2, width="stretch")

    st.markdown("### 3. 漂移布朗运动：期望值")
    st.latex(r"\mathbb{E}[X(t)] = \mu t")

    X = sim_drift_bm_3d(n_particles, steps, dt, mu, sigma, seed + 1)
    mean_sim = X[:, :, 0].mean(axis=1)
    mean_theory = mu * t

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=t, y=mean_sim, mode="lines", name="模拟均值"))
    fig3.add_trace(go.Scatter(x=t, y=mean_theory, mode="lines",
                              name="理论均值",
                              line=dict(dash="dash")))
    fig3.update_layout(title="漂移布朗运动：期望", xaxis_title="t", yaxis_title="均值")

    st.plotly_chart(fig3, width="stretch")

    st.markdown("### 4. 相关布朗运动：协方差验证")
    st.latex(r"\mathrm{Cov}(X_i(t), X_j(t)) = \sigma^2 t\,\rho_{ij}")

    X = sim_correlated_bm_3d(n_particles, steps, dt, sigma, rho, seed + 2)
    X_t = X[-1]

    cov_sim = np.cov(X_t.T)
    cov_theory = sigma**2 * t[-1] * _spd_corr_matrix(3, rho)

    fig_cov = go.Figure()
    fig_cov.add_trace(go.Heatmap(z=cov_sim, colorbar=dict(title="协方差")))
    fig_cov.update_layout(title="相关布朗运动：模拟协方差矩阵")

    fig_cov_th = go.Figure()
    fig_cov_th.add_trace(go.Heatmap(z=cov_theory, colorbar=dict(title="协方差")))
    fig_cov_th.update_layout(title="相关布朗运动：理论协方差矩阵")

    st.plotly_chart(fig_cov, width="stretch")
    st.plotly_chart(fig_cov_th, width="stretch")

    st.markdown("### 5. 几何布朗运动：期望值")
    st.latex(r"\mathbb{E}[X(t)] = X_0 e^{\mu t}")

    X = sim_gbm_3d(n_particles, steps, dt, mu, sigma, rho, x0, seed + 3)
    mean_sim = X[:, :, 0].mean(axis=1)
    mean_theory = x0 * np.exp(mu * t)

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=t, y=mean_sim, mode="lines", name="模拟均值"))
    fig4.add_trace(go.Scatter(x=t, y=mean_theory, mode="lines",
                              name="理论期望",
                              line=dict(dash="dash")))
    fig4.update_layout(title="几何布朗运动：期望", xaxis_title="t", yaxis_title="均值")

    st.plotly_chart(fig4, width="stretch")


st.divider()
st.markdown(
    r"**说明**：左侧调整参数 $\mu,\sigma,\rho$，观察不同布朗运动的随机行为差异。"
)
