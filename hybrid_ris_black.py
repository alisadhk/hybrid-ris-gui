# -*- coding: utf-8 -*-
"""
Hybrid Active–Passive RIS — DRL Visualization GUI
Dark mode + default plot colors restored + non-intrusive Qt tooltip
(Core math unchanged)
"""
import sys, math, random
from dataclasses import dataclass
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import rcParams

# ---------- Dark theme with WHITE texts; use DEFAULT line colors ----------
rcParams.update({
    "figure.constrained_layout.use": True,
    "figure.facecolor": "#1C2D47",
    "savefig.facecolor": "#263A58",
    "axes.facecolor": "#0E1013",
    "axes.edgecolor": "#2A2F35",
    "axes.labelcolor": "#e6e6e6",
    "text.color": "#e6e6e6",          # white text
    "xtick.color": "#cfd6de",
    "ytick.color": "#cfd6de",
    "grid.color": "#3a414b",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "axes.grid.which": "both",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": True,
    "legend.framealpha": 0.15,
    "legend.fancybox": True,
    "lines.linewidth": 2.2,
    "lines.markersize": 6.5,
    "font.size": 10.5,
    # Note: axes.prop_cycle is not set -> default matplotlib colors are used (as before)
})

# ---------------- data classes (unchanged) ----------------
@dataclass
class StaticSettings:
    bs_xyz: tuple = (0.0, 0.0, 0.0)
    ris_xyz: tuple = (0.0, 10.0, 50.0)
    user_xyz: tuple = (10.0, 10.0, 0.0)
    eve_xyz: tuple  = (30.0, 0.0, 0.0)
    jam_xyz: tuple  = (0.0, 30.0, 0.0)
    lr_actor: float = 0.001
    lr_critic: float = 0.001
    gamma: float = 0.99
    batch_size: int = 64
    replay_buffer_size: int = 1_000_000
    tau: float = 0.005
    max_episodes: int = 2000
    max_steps_per_episode: int = 1000

@dataclass
class MainParams:
    M: int = 12
    L: int = 6
    N: int = 150
    N_a: int = 30
    P_B_max: float = 35.0
    P_J_max: float = 25.0
    sigma2: float = 1e-12
    alpha0: float = -30.0
    mu: float = 2.7
    K_factor: float = 3.0

# ---------------- kernels (unchanged) ----------------
def _db_to_linear(db): return 10 ** (db / 10.0)

def secrecy_vs_active_elements(p: MainParams):
    import math
    max_show = min(p.N, max(p.N_a, 10))
    xs = list(range(5, min(36, max_show + 6), 5))
    pb = _db_to_linear(p.P_B_max)
    base = math.log10(1 + pb) * (1 + 0.1 * math.log2(1 + p.M))
    ddpg, ao = [], []
    for n in xs:
        gain = base * (1.0 - math.exp(-n / 12.0)) + 0.5 * math.log2(1 + p.K_factor)
        ddpg.append(0.8 + 0.35 * n / 5 + 0.08 * math.sqrt(max(n - 4, 0)) + 0.25 * math.log2(1 + gain))
        ao.append  (0.4 + 0.22 * n / 5 + 0.04 * math.sqrt(max(n - 4, 0)) + 0.18 * math.log2(1 + gain))
    return xs, ddpg, ao

def reward_vs_episodes_curves(s: StaticSettings):
    import math, random
    episodes = list(range(1, s.max_episodes + 1, 10))
    lrs = [0.01, 0.001, 0.0001, 0.00001]
    curves = {}
    for lr in lrs:
        target = 18.0 - 4.0 * (math.log10(0.01/lr) if lr <= 0.01 else 0)
        k = 0.025 + 0.01 * math.log10(1.0/lr)
        ys, avg = [], 0.0
        for ep in episodes:
            avg = target * (1.0 - math.exp(-k * ep))
            noise = (random.random() - 0.5) * 0.6
            ys.append(max(0.0, avg + noise))
        curves[lr] = ys
    return episodes, curves

def secrecy_vs_total_power(p: MainParams):
    import math
    xs = list(range(10, 31, 2))
    passive, hyb0, hyb10, active10 = [], [], [], []
    for pt in xs:
        base = 0.1 * (pt - 10) + 1.0 + 0.1 * math.log2(1 + p.M) + 0.05 * math.log2(1 + p.N/50)
        passive.append (base * 0.6  + 0.2)
        hyb0.append    (base * 0.85 + 0.5)
        hyb10.append   (base * 0.95 + 0.7)
        active10.append(base * 1.05 + 0.9)
    return xs, hyb0, hyb10, active10, passive

def secrecy_vs_Na_multi(p: MainParams):
    import math
    results = {}
    ptots, Ns = [20, 30], [50, 100, 150, 200]
    xs = [1, 2, 3, 4, 6, 8, 10]
    for pt in ptots:
        for n in Ns:
            ys = []
            for a in xs:
                base = 7.0 + 0.12*(pt-20) + 0.12*math.log2(1 + n/50) + 0.22*math.log2(1 + p.M/8)
                val = base + 0.5*(1 - math.exp(-(a)/3.0)) + 0.05*math.sqrt(max(a-1,0))
                ys.append(val)
            results[(pt, n)] = (xs, ys)
    return results

# ---------------- Matplotlib widget + Qt Tooltip (does not change the drawing) ----------------
def _beautify_axes(ax, xlab, ylab, title=None, legend_kw=None):
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    if title: ax.set_title(title, color="#e8e8e6", pad=8)
    ax.tick_params(axis='both', which='major', length=6, width=1.2)
    ax.tick_params(axis='both', which='minor', length=3, width=0.8)
    ax.margins(x=0.02)
    if legend_kw is None: legend_kw = {}
    leg = ax.legend(**({"fontsize": 9, "loc": "best", "ncol": 1, **legend_kw}))
    if leg:
        leg.get_frame().set_edgecolor("#6b7785")
        for t in leg.get_texts(): t.set_color("#e6e6e6")

class PlotCanvas(QtWidgets.QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0); lay.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(title, color="#e8e8e6")

        # Qt text tooltip (does not change axes/drawing)
        self._tooltip_text = ""
        self._cid_motion = None
        self._cid_leave = None

    def clear(self):
        self.figure.clf()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title(self.ax.get_title(), color="#e8e8e6")
        # keep the tooltip
        self._install_qtooltip(self._tooltip_text)

    def draw(self): self.canvas.draw_idle()

    def _on_motion(self, event):
        if not self._tooltip_text: return
        # cursor position at QWidget level (no drawing change)
        if event is None: return
        global_pt = self.canvas.mapToGlobal(QtCore.QPoint(int(event.x), int(event.y)))
        QtWidgets.QToolTip.showText(global_pt, self._tooltip_text, self.canvas)

    def _on_leave(self, event):
        QtWidgets.QToolTip.hideText()

    def _install_qtooltip(self, text: str):
        self._tooltip_text = text or ""
        if self._cid_motion is not None:
            self.canvas.mpl_disconnect(self._cid_motion); self._cid_motion = None
        if self._cid_leave is not None:
            self.canvas.mpl_disconnect(self._cid_leave); self._cid_leave = None
        if self._tooltip_text:
            self._cid_motion = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self._cid_leave  = self.canvas.mpl_connect("figure_leave_event", self._on_leave)

    def set_hover_text(self, text):
        self._install_qtooltip(text)

# ---------------- Main Window ----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hybrid Active–Passive RIS — DRL Results GUI")
        self.resize(1200, 800)
        self.settings, self.params = StaticSettings(), MainParams()

        QtWidgets.QApplication.setStyle("Fusion")
        self._apply_dark_palette()

        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        hsplit = QtWidgets.QHBoxLayout(central)

        # --------- LEFT ---------
        left = QtWidgets.QFrame(); left.setFrameShape(QtWidgets.QFrame.StyledPanel)
        left_v = QtWidgets.QVBoxLayout(left); left_v.setContentsMargins(10,10,10,10); left_v.setSpacing(10)

        grp_inputs = QtWidgets.QGroupBox("Main Params")
        form = QtWidgets.QFormLayout(grp_inputs)
        self.spin_M = QtWidgets.QSpinBox(); self.spin_M.setRange(1, 128); self.spin_M.setValue(self.params.M)
        self.spin_L = QtWidgets.QSpinBox(); self.spin_L.setRange(1, 64); self.spin_L.setValue(self.params.L)
        self.spin_N = QtWidgets.QSpinBox(); self.spin_N.setRange(10, 10000); self.spin_N.setValue(self.params.N)
        self.spin_Na= QtWidgets.QSpinBox(); self.spin_Na.setRange(0, 10000); self.spin_Na.setValue(self.params.N_a)
        self.dsb_PB = QtWidgets.QDoubleSpinBox(); self.dsb_PB.setRange(0.0, 100.0); self.dsb_PB.setDecimals(3); self.dsb_PB.setValue(self.params.P_B_max)
        self.dsb_PJ = QtWidgets.QDoubleSpinBox(); self.dsb_PJ.setRange(0.0, 100.0); self.dsb_PJ.setDecimals(3); self.dsb_PJ.setValue(self.params.P_J_max)
        self.dsb_sigma2 = QtWidgets.QDoubleSpinBox(); self.dsb_sigma2.setRange(1e-15, 1e-6); self.dsb_sigma2.setDecimals(15); self.dsb_sigma2.setSingleStep(1e-13); self.dsb_sigma2.setValue(self.params.sigma2)
        form.addRow("M (BS antennas):", self.spin_M)
        form.addRow("L (Jammer antennas):", self.spin_L)
        form.addRow("N (Total RIS):", self.spin_N)
        form.addRow("N_a (Active RIS):", self.spin_Na)
        form.addRow("P_B_max (dBm):", self.dsb_PB)
        form.addRow("P_J_max (dBm):", self.dsb_PJ)
        form.addRow("sigma^2 (noise):", self.dsb_sigma2)

        self.dsb_alpha0 = QtWidgets.QDoubleSpinBox(); self.dsb_alpha0.setRange(-200.0, 0.0); self.dsb_alpha0.setDecimals(3); self.dsb_alpha0.setValue(self.params.alpha0)
        self.dsb_mu = QtWidgets.QDoubleSpinBox(); self.dsb_mu.setRange(1.0, 6.0); self.dsb_mu.setDecimals(3); self.dsb_mu.setValue(self.params.mu)
        self.dsb_K = QtWidgets.QDoubleSpinBox(); self.dsb_K.setRange(0.0, 50.0); self.dsb_K.setDecimals(3); self.dsb_K.setValue(self.params.K_factor)
        form.addRow("alpha0 (dB @1m):", self.dsb_alpha0)
        form.addRow("mu (path-loss exp):", self.dsb_mu)
        form.addRow("K-factor (Rician):", self.dsb_K)
        left_v.addWidget(grp_inputs)

        grp_set = QtWidgets.QGroupBox("Settings")
        fs = QtWidgets.QFormLayout(grp_set)
        self.bs_edit   = QtWidgets.QLineEdit(str(self.settings.bs_xyz))
        self.ris_edit  = QtWidgets.QLineEdit(str(self.settings.ris_xyz))
        self.user_edit = QtWidgets.QLineEdit(str(self.settings.user_xyz))
        self.eve_edit  = QtWidgets.QLineEdit(str(self.settings.eve_xyz))
        self.jam_edit  = QtWidgets.QLineEdit(str(self.settings.jam_xyz))
        for w in (self.bs_edit, self.ris_edit, self.user_edit, self.eve_edit, self.jam_edit):
            w.setPlaceholderText("(x, y, z)")
        fs.addRow("BS (x,y,z):", self.bs_edit)
        fs.addRow("RIS (x,y,z):", self.ris_edit)
        fs.addRow("User (x,y,z):", self.user_edit)
        fs.addRow("Eavesdropper (x,y,z):", self.eve_edit)
        fs.addRow("Jammer (x,y,z):", self.jam_edit)

        self.lr_actor = QtWidgets.QDoubleSpinBox(); self.lr_actor.setDecimals(6); self.lr_actor.setRange(1e-6,1.0); self.lr_actor.setValue(self.settings.lr_actor)
        self.lr_critic= QtWidgets.QDoubleSpinBox(); self.lr_critic.setDecimals(6); self.lr_critic.setRange(1e-6,1.0); self.lr_critic.setValue(self.settings.lr_critic)
        self.gamma    = QtWidgets.QDoubleSpinBox(); self.gamma.setDecimals(4); self.gamma.setRange(0.0,0.9999); self.gamma.setSingleStep(0.0001); self.gamma.setValue(self.settings.gamma)
        self.batch    = QtWidgets.QSpinBox(); self.batch.setRange(1,1_000_000); self.batch.setValue(self.settings.batch_size)
        self.replay   = QtWidgets.QSpinBox(); self.replay.setRange(1,10_000_000); self.replay.setValue(self.settings.replay_buffer_size)
        self.tau      = QtWidgets.QDoubleSpinBox(); self.tau.setDecimals(6); self.tau.setRange(1e-6,1.0); self.tau.setValue(self.settings.tau)
        self.max_ep   = QtWidgets.QSpinBox(); self.max_ep.setRange(1,100_000); self.max_ep.setValue(self.settings.max_episodes)
        self.max_steps= QtWidgets.QSpinBox(); self.max_steps.setRange(1,1_000_000); self.max_steps.setValue(self.settings.max_steps_per_episode)
        fs.addRow("LR (Actor):", self.lr_actor)
        fs.addRow("LR (Critic):", self.lr_critic)
        fs.addRow("Discount γ:", self.gamma)
        fs.addRow("Batch Size:", self.batch)
        fs.addRow("Replay Buffer Size:", self.replay)
        fs.addRow("Target Update τ:", self.tau)
        fs.addRow("Max Episodes:", self.max_ep)
        fs.addRow("Max Steps/Episode:", self.max_steps)
        left_v.addWidget(grp_set)

        self.btn_run = QtWidgets.QPushButton("- Run -")
        self.btn_run.setMinimumHeight(38)
        left_v.addWidget(self.btn_run)
        left_v.addStretch(1)

        # --------- RIGHT ---------
        right = QtWidgets.QWidget(); grid = QtWidgets.QGridLayout(right)
        grid.setContentsMargins(6,6,6,6); grid.setSpacing(8)
        self.plot1 = PlotCanvas("Secrecy vs Number of Hybrid-RIS elements")
        self.plot2 = PlotCanvas("Average Reward vs Episodes (various α)")
        self.plot3 = PlotCanvas("Secrecy vs Total Transmit Power")
        self.plot4 = PlotCanvas("Secrecy vs Number of Active Elements (multi N, Ptot)")
        grid.addWidget(self.plot1, 0, 0); grid.addWidget(self.plot2, 0, 1)
        grid.addWidget(self.plot3, 1, 0); grid.addWidget(self.plot4, 1, 1)
        hsplit.addWidget(left, 0); hsplit.addWidget(right, 1)

        self.btn_run.clicked.connect(self.on_run)

    # ---- Dark palette for Qt ----
    def _apply_dark_palette(self):
        pal = QPalette()
        base = QColor(17,19,21); alt = QColor(22,25,28); text = QColor(230,230,230)
        pal.setColor(QPalette.Window, QColor(13,15,18))
        pal.setColor(QPalette.WindowText, text)
        pal.setColor(QPalette.Base, base)
        pal.setColor(QPalette.AlternateBase, alt)
        pal.setColor(QPalette.ToolTipBase, base)
        pal.setColor(QPalette.ToolTipText, text)
        pal.setColor(QPalette.Text, text)
        pal.setColor(QPalette.Button, alt)
        pal.setColor(QPalette.ButtonText, text)
        pal.setColor(QPalette.BrightText, QColor(255,84,84))
        pal.setColor(QPalette.Highlight, QColor(0,120,215))
        pal.setColor(QPalette.HighlightedText, QColor(255,255,255))
        pal.setColor(QPalette.PlaceholderText, QColor(170,170,170))
        QtWidgets.QApplication.setPalette(pal)
        QtWidgets.QApplication.instance().setStyleSheet("""
        QGroupBox{
            border:1px solid #2a2f36;border-radius:10px;margin-top:12px;
            padding:8px;color:#e6e6e6;
        }
        QGroupBox::title{left:10px;color:#cfd6de;}
        QPushButton{
            border:1px solid #2a2f36;border-radius:10px;padding:8px 14px;
            background:#1a1d21;color:#e6e6e6;
        }
        QPushButton:hover{background:#20252b;}
        QPushButton:pressed{background:#111315;}
        QLineEdit,QSpinBox,QDoubleSpinBox{
            border:1px solid #2a2f36;border-radius:8px;padding:4px 6px;
            background:#121418;selection-background-color:#0078d7;selection-color:#fff;
        }""")

    # ---------------- sync helpers ----------------
    def _parse_tuple(self, t):
        try:
            v = eval(t, {}, {})
            if isinstance(v,(list,tuple)) and len(v)==3: return (float(v[0]),float(v[1]),float(v[2]))
        except Exception: pass
        return None

    def _sync_params(self):
        p = self.params
        p.M=int(self.spin_M.value()); p.L=int(self.spin_L.value()); p.N=int(self.spin_N.value())
        p.N_a=int(self.spin_Na.value()); p.P_B_max=float(self.dsb_PB.value()); p.P_J_max=float(self.dsb_PJ.value())
        p.sigma2=float(self.dsb_sigma2.value()); p.alpha0=float(self.dsb_alpha0.value()); p.mu=float(self.dsb_mu.value()); p.K_factor=float(self.dsb_K.value())

    def _sync_settings(self):
        s = self.settings
        bs=self._parse_tuple(self.bs_edit.text()); ris=self._parse_tuple(self.ris_edit.text())
        user=self._parse_tuple(self.user_edit.text()); eve=self._parse_tuple(self.eve_edit.text()); jam=self._parse_tuple(self.jam_edit.text())
        if not all([bs,ris,user,eve,jam]):
            QtWidgets.QMessageBox.warning(self,"Invalid","Please make the correct changes such as: (0, 10, 50)"); return False
        s.bs_xyz=bs; s.ris_xyz=ris; s.user_xyz=user; s.eve_xyz=eve; s.jam_xyz=jam
        s.lr_actor=float(self.lr_actor.value()); s.lr_critic=float(self.lr_critic.value())
        s.gamma=float(self.gamma.value()); s.batch_size=int(self.batch.value())
        s.replay_buffer_size=int(self.replay.value()); s.tau=float(self.tau.value())
        s.max_episodes=int(self.max_ep.value()); s.max_steps_per_episode=int(self.max_steps.value())
        return True

    # ---------------- actions ----------------
    def on_run(self):
        self._sync_params()
        if not self._sync_settings(): return

        # Plot 1
        x1, ddpg, ao = secrecy_vs_active_elements(self.params)
        self.plot1.clear()
        self.plot1.ax.plot(x1, ao, marker='s', label="AO")
        self.plot1.ax.plot(x1, ddpg, marker='o', label="Proposed DDPG")
        _beautify_axes(self.plot1.ax, "Number of Hybrid-RIS elements", "Secrecy rate [bits/s/Hz]",
                       title="Secrecy vs Number of Hybrid-RIS elements")
        self.plot1.set_hover_text(
            "AO:   R = 0.4 + 0.22*(n/5) + 0.04*sqrt(max(n-4,0)) + 0.18*log2(1+gain)\n"
            "DDPG: R = 0.8 + 0.35*(n/5) + 0.08*sqrt(max(n-4,0)) + 0.25*log2(1+gain)\n"
            "gain = base*(1 - e^{-n/12}) + 0.5*log2(1+K), base = log10(1+10^{P_B/10})*(1+0.1*log2(1+M))"
        )
        self.plot1.draw()

        # Plot 2
        ep, curves = reward_vs_episodes_curves(self.settings)
        self.plot2.clear()
        for lr, ys in curves.items():
            self.plot2.ax.plot(ep, ys, marker='.', label=f"Learning rate = {lr}")
        _beautify_axes(self.plot2.ax, "Episodes", "Average reward",
                       title="Average Reward vs Episodes (various α)")
        self.plot2.set_hover_text(
            "avg = target*(1 - e^{-k*episode})\n"
            "target = 18 - 4*log10(0.01/lr),  k = 0.025 + 0.01*log10(1/lr)"
        )
        self.plot2.draw()

        # Plot 3
        x3, hyb0, hyb10, active10, passive = secrecy_vs_total_power(self.params)
        self.plot3.clear()
        self.plot3.ax.plot(x3, hyb0, marker='o', label="Hybrid RIS, Pa_max = 0 dBm")
        self.plot3.ax.plot(x3, hyb10, marker='^', label="Hybrid RIS, Pa_max = 10 dBm")
        self.plot3.ax.plot(x3, active10, marker='D', label="Active RIS, Pa_max = 10 dBm")
        self.plot3.ax.plot(x3, passive, linestyle='-.', label="Passive RIS")
        _beautify_axes(self.plot3.ax, "Total transmit power (P_tot) [dBm]",
                       "Secrecy rate (R) [bits/s/Hz]",
                       title="Secrecy vs Total Transmit Power")
        self.plot3.set_hover_text(
            "base = 0.1*(P_tot-10) + 1 + 0.1*log2(1+M) + 0.05*log2(1+N/50)\n"
            "R_passive = 0.6*base + 0.2 | R_hybrid0 = 0.85*base + 0.5\n"
            "R_hybrid10 = 0.95*base + 0.7 | R_active10 = 1.05*base + 0.9"
        )
        self.plot3.draw()

        # Plot 4
        self.plot4.clear()
        multi = secrecy_vs_Na_multi(self.params)
        for (p, n), (xs, ys) in sorted(multi.items()):
            self.plot4.ax.plot(xs, ys, marker='o', label=f"P_tot={p} dBm, N={n}")
        _beautify_axes(self.plot4.ax, "Number of active elements (N_a)",
                       "Secrecy rate [bits/s/Hz]",
                       title="Secrecy vs Number of Active Elements (multi N, Ptot)",
                       legend_kw={"fontsize": 8, "ncol": 2})
        self.plot4.set_hover_text(
            "R = base + 0.5*(1 - e^{-N_a/3}) + 0.05*sqrt(max(N_a-1,0))\n"
            "base = 7 + 0.12*(P_tot-20) + 0.12*log2(1+N/50) + 0.22*log2(1+M/8)"
        )
        self.plot4.draw()

# ---------------- app entry ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
