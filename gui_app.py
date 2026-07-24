
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
import numpy as np

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from environment import Environment, decode_state
from parameters import (
    data_queue_capacity,
    energy_queue_capacity,
    total_actions,
    results_directory,
    models_directory,
    total_states,
    q_learning_training_steps,
    dqn_training_steps,
    initial_exploration_rate,
    minimum_exploration_rate,
    exploration_decay_rate,
    target_network_update_frequency,
    active_transmit_energy_per_packet,
)
from utils import get_action_name

BG_ROOT = "#f0f4f8"
BG_CARD = "#ffffff"
BG_CANVAS = "#f7fafc"
BORDER = "#d1d9e6"
MUTED = "#64748b"
TEXT_DARK = "#0f172a"
TEXT_LABEL = "#475569"

ACCENT_BLUE = "#2563eb"
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"
ACCENT_ORANGE = "#f97316"
ACCENT_AMBER = "#f59e0b"
ACCENT_VIOLET = "#8b5cf6"

JAMMER_IDLE_CLR = "#64748b"
JAMMER_IDLE_GLOW = "#f1f5f9"

BTN_BLUE = "#2563eb"
BTN_GREEN = "#10b981"
BTN_VIOLET = "#8b5cf6"
BTN_GREY = "#475569"
BTN_ORANGE = "#f97316"
BTN_DANGER = "#ef4444"
BTN_DISABLED = "#94a3b8"
BTN_HOVER = "#1e40af"

MANUAL_ACTION_SPECS = [
    {"label": "Idle", "shortcut": "0", "color": BTN_GREY, "group": "idle",
     "desc": "Do nothing — jammer transitions naturally (always available)"},
    {"label": "Active TX", "shortcut": "1", "color": BTN_BLUE, "group": "idle",
     "desc": "Transmit when jammer is IDLE — needs data and energy"},
    {"label": "Harvest", "shortcut": "2", "color": BTN_GREEN, "group": "active",
     "desc": "Harvest RF energy when jammer is ACTIVE"},
    {"label": "Backscatter", "shortcut": "3", "color": BTN_ORANGE, "group": "active",
     "desc": "Reflect jammer signal to send data — needs data queue"},
    {"label": "RA-0", "shortcut": "4", "color": ACCENT_VIOLET, "group": "active",
     "desc": "Rate adaptation level 0 — highest throughput when jammer active"},
    {"label": "RA-1", "shortcut": "5", "color": ACCENT_VIOLET, "group": "active",
     "desc": "Rate adaptation level 1 — medium throughput"},
    {"label": "RA-2", "shortcut": "6", "color": ACCENT_VIOLET, "group": "active",
     "desc": "Rate adaptation level 2 — lowest throughput, most robust"},
]

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#1e293b", foreground="#f1f5f9",
                         relief=tk.SOLID, borderwidth=1,
                         font=("Segoe UI", 9), padx=10, pady=8, wraplength=400)
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        tw = self.tooltip_window
        self.tooltip_window = None
        if tw:
            tw.destroy()

class AmbientJammingGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ambient Backscatter Anti-Jamming — RL Visualizer")
        self.root.geometry("1300x850")
        self.root.configure(bg=BG_ROOT)
        self.root.resizable(True, True)

        self.environment = Environment(seed=None)

        self.q_table = None
        self.q_table_path = None
        self.load_q_table()

        self.dqn_model = None
        self.dqn_model_path = None
        self.load_dqn_model()

        self.running = False
        self.training_in_progress = False
        self.step_delay = 700
        self.total_steps = 0
        self.total_packets = 0
        self.last_action = 0
        self.last_reward = 0
        self.animation_after_id = None
        self._previous_policy = None

        self.transmitter_position = (0, 0)
        self.receiver_position = (0, 0)
        self.jammer_position = (0, 0)

        self.policy_names = {
            "Q-Learning": "q",
            "Deep Q-Network (DQN)": "dqn",
            "Random": "random",
            "Manual": "manual"
        }
        self.policy_names_reverse = {v: k for k, v in self.policy_names.items()}
        
        initial_abbreviation = "random"
        if self.q_table is not None:
            initial_abbreviation = "q"
        elif self.dqn_model is not None:
            initial_abbreviation = "dqn"
        self.policy_mode = tk.StringVar(value=self.policy_names_reverse[initial_abbreviation])

        self.build_ui()
        self.root.update_idletasks()
        self.draw_scene()

    def load_q_table(self):
        # Try models directory first, then results directory as fallback
        primary_path = os.path.join(models_directory, "q_table.npy")
        fallback_path = os.path.join(results_directory, "q_table.npy")
        self.q_table_path = primary_path if os.path.exists(primary_path) else fallback_path
        if os.path.exists(self.q_table_path):
            try:
                self.q_table = np.load(self.q_table_path)
            except Exception as error:
                print(f"[GUI] Q-table load error: {error}")

    def load_dqn_model(self):
        # Try models directory first, then results directory as fallback
        primary_path = os.path.join(models_directory, "dqn_model.keras")
        fallback_path = os.path.join(results_directory, "dqn_model.keras")
        self.dqn_model_path = primary_path if os.path.exists(primary_path) else fallback_path
        if os.path.exists(self.dqn_model_path):
            try:
                import tensorflow as tf
                self.dqn_model = tf.keras.models.load_model(self.dqn_model_path)
            except Exception as error:
                print(f"[GUI] DQN load error: {error}")

    def build_ui(self):
        header = tk.Frame(self.root, bg=BG_ROOT)
        header.pack(fill=tk.X, padx=32, pady=(28, 0))

        tk.Label(header,
                 text="Ambient Backscatter Anti-Jamming",
                 font=("Segoe UI", 24, "bold"), fg=TEXT_DARK, bg=BG_ROOT
                 ).pack(side=tk.LEFT)

        tk.Label(header,
                 text="Reinforcement Learning Visualizer",
                 font=("Segoe UI", 13), fg=TEXT_LABEL, bg=BG_ROOT
                 ).pack(side=tk.LEFT, padx=(20, 0), pady=(6, 0))

        tk.Frame(self.root, bg=ACCENT_BLUE, height=3).pack(fill=tk.X, padx=32, pady=(10, 0))

        body = tk.Frame(self.root, bg=BG_ROOT)
        body.pack(fill=tk.BOTH, expand=True, padx=32, pady=20)

        canvas_card = tk.Frame(body, bg=BG_CARD, bd=0, relief="flat",
                               highlightthickness=1, highlightbackground=BORDER)
        canvas_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_card, bg=BG_CANVAS,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda event: self.draw_scene())

        side = tk.Frame(body, bg=BG_ROOT, width=260)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        side.pack_propagate(False)
        self.build_side_panel(side)

        self.build_stats_bar()

        self.build_console()

        self.build_controls()

    def build_side_panel(self, parent):
        self.make_section_label(parent, "Jammer Status")
        jammer_card = tk.Frame(parent, bg=BG_CARD, bd=0, relief="flat",
                            highlightthickness=1, highlightbackground=BORDER)
        jammer_card.pack(fill=tk.X, pady=(4, 14), padx=0)

        self.jammer_label = tk.Label(jammer_card, text="● IDLE",
                                     font=("Segoe UI", 16, "bold"),
                                     fg=BTN_GREY, bg=BG_CARD, pady=16)
        self.jammer_label.pack()

        self.make_section_label(parent, "Data Queue")
        self.data_meter = self.make_meter(parent)

        self.make_section_label(parent, "Energy Queue")
        self.energy_meter = self.make_meter(parent)

        self.make_section_label(parent, "Last Action")
        action_card = tk.Frame(parent, bg=BG_CARD, bd=0, relief="flat",
                             highlightthickness=1, highlightbackground=BORDER)
        action_card.pack(fill=tk.X, pady=(4, 14), padx=0)

        self.action_label = tk.Label(action_card, text="—",
                                     font=("Segoe UI", 11, "bold"),
                                     fg=ACCENT_BLUE, bg=BG_CARD, wraplength=240,
                                     pady=12, padx=14, justify=tk.LEFT)
        self.action_label.pack(anchor=tk.W)

        self.reward_label = tk.Label(action_card, text="Reward: —",
                                     font=("Segoe UI", 10), fg=MUTED,
                                     bg=BG_CARD, pady=6, padx=14)
        self.reward_label.pack(anchor=tk.W)

        self.make_section_label(parent, "Simulation Speed")
        speed_card = tk.Frame(parent, bg=BG_CARD, bd=0, relief="flat",
                              highlightthickness=1, highlightbackground=BORDER)
        speed_card.pack(fill=tk.X, pady=(4, 14), padx=0)

        self.speed_variable = tk.IntVar(value=self.step_delay)
        speed_scale = tk.Scale(speed_card, from_=200, to=2000, orient=tk.HORIZONTAL,
                               variable=self.speed_variable, resolution=100,
                               command=self.on_speed_change,
                               bg=BG_CARD, fg=TEXT_DARK, troughcolor=BORDER,
                               highlightthickness=0, bd=0, sliderrelief="flat",
                               sliderlength=20, length=220, label="Step delay (ms)",
                               font=("Segoe UI", 9))
        speed_scale.pack(padx=12, pady=10)

    def make_section_label(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 8, "bold"),
                 fg=TEXT_LABEL, bg=BG_ROOT, anchor=tk.W
                 ).pack(fill=tk.X, pady=(12, 2))

    def make_meter(self, parent):
        canvas = tk.Canvas(parent, bg=BG_CARD, height=36, highlightthickness=1,
                          highlightbackground=BORDER)
        canvas.pack(fill=tk.X, pady=(4, 0))
        return canvas

    def draw_meter(self, canvas_widget, value, capacity, color):
        canvas_widget.delete("all")
        width = canvas_widget.winfo_width() or 240
        height = canvas_widget.winfo_height() or 36
        segment_gap = 4
        total_gap = segment_gap * (capacity - 1)
        segment_width = (width - 20 - total_gap) / capacity
        x = 10
        for i in range(capacity):
            fill = color if i < value else "#e2e8f0"
            canvas_widget.create_rectangle(x, 8, x + segment_width, height - 8,
                                           fill=fill, outline="", width=0)
            x += segment_width + segment_gap
        canvas_widget.create_text(width // 2, height // 2,
                                  text=f"{value} / {capacity}",
                                  font=("Segoe UI", 9, "bold"),
                                  fill=TEXT_DARK if value > 0 else MUTED)

    def build_stats_bar(self):
        bar = tk.Frame(self.root, bg=BG_CARD, bd=0,
                       highlightthickness=1, highlightbackground=BORDER)
        bar.pack(fill=tk.X, padx=32, pady=(0, 10))

        stats = [
            ("Timesteps", "steps_label", "0"),
            ("Packets Delivered", "packets_label", "0"),
            ("Average Throughput", "average_label", "0.0000 packets/slot"),
            ("Current State", "state_label", "(j=0, d=0, e=0)"),
        ]
        for i, (title, attribute, initial) in enumerate(stats):
            cell = tk.Frame(bar, bg=BG_CARD)
            cell.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
            tk.Label(cell, text=title, font=("Segoe UI", 8, "bold"),
                     fg=TEXT_LABEL, bg=BG_CARD).pack(pady=(10, 2))
            label = tk.Label(cell, text=initial, font=("Segoe UI", 14, "bold"),
                           fg=TEXT_DARK, bg=BG_CARD)
            label.pack(pady=(2, 10))
            setattr(self, attribute, label)
            if i < len(stats) - 1:
                tk.Frame(bar, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y,
                                                        pady=10)

    def build_console(self):
        console_frame = tk.Frame(self.root, bg=BG_ROOT)
        console_frame.pack(fill=tk.X, padx=32, pady=(0, 8))

        tk.Label(console_frame, text="Simulation Log",
                 font=("Segoe UI", 8, "bold"), fg=TEXT_LABEL, bg=BG_ROOT
                 ).pack(anchor=tk.W, pady=(0, 4))

        log_background = tk.Frame(console_frame, bg=BG_CARD, bd=0,
                                  highlightthickness=1, highlightbackground=BORDER)
        log_background.pack(fill=tk.X)

        self.log_text = tk.Text(log_background, bg=BG_CARD, fg=TEXT_DARK,
                                font=("Consolas", 9), height=5,
                                state=tk.DISABLED, bd=0, wrap=tk.WORD,
                                selectbackground="#dbeafe",
                                highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, side=tk.LEFT, expand=True, padx=14, pady=10)

        self.log_text.tag_config("header", foreground=ACCENT_BLUE, font=("Consolas", 9, "bold"))
        self.log_text.tag_config("reward", foreground=ACCENT_GREEN, font=("Consolas", 9, "bold"))
        self.log_text.tag_config("warn", foreground=ACCENT_RED, font=("Consolas", 9, "bold"))
        self.log_text.tag_config("train", foreground=ACCENT_VIOLET, font=("Consolas", 9, "bold"))

        scrollbar = tk.Scrollbar(log_background, command=self.log_text.yview,
                          bg=BG_CARD, troughcolor=BG_CARD, bd=0, width=14)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def build_controls(self):
        controls = tk.Frame(self.root, bg=BG_CARD, bd=0,
                          highlightthickness=1, highlightbackground=BORDER)
        controls.pack(fill=tk.X, padx=32, pady=(0, 20))

        top_row = tk.Frame(controls, bg=BG_CARD)
        top_row.pack(fill=tk.X, padx=18, pady=(14, 10))

        policy_group = tk.Frame(top_row, bg=BG_CARD)
        policy_group.pack(side=tk.LEFT, padx=(0, 22))
        tk.Label(policy_group, text="Policy", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_LABEL, bg=BG_CARD).pack(anchor=tk.W)
        
        self.mode_selector = ttk.Combobox(
            policy_group, textvariable=self.policy_mode,
            values=list(self.policy_names.keys()),
            state="readonly", width=30, font=("Segoe UI", 10)
        )
        self.mode_selector.pack(pady=(2, 0))
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', 
                        fieldbackground=BG_CARD,
                        background=BG_CARD,
                        foreground=TEXT_DARK,
                        arrowcolor=TEXT_DARK,
                        borderwidth=1,
                        relief='flat',
                        font=("Segoe UI", 10))
        style.map('TCombobox',
                 fieldbackground=[('readonly', BG_CARD)],
                 background=[('readonly', BG_CARD)],
                 foreground=[('readonly', TEXT_DARK)])

        tk.Frame(top_row, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y,
                                                    padx=(0, 20), pady=2)

        simulation_group = tk.Frame(top_row, bg=BG_CARD)
        simulation_group.pack(side=tk.LEFT, padx=(0, 20))
        tk.Label(simulation_group, text="Simulation", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_LABEL, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 4))
        btn_row = tk.Frame(simulation_group, bg=BG_CARD)
        btn_row.pack()
        self.play_button = self.make_button(btn_row, "▶ Run", self.toggle_run, BTN_BLUE, width=10)
        self.play_button.pack(side=tk.LEFT, padx=3)
        self.make_button(btn_row, "⟳ Step", self.simulation_step, BTN_GREY, width=10).pack(side=tk.LEFT, padx=3)
        self.make_button(btn_row, "⟲ Reset", self.reset_simulation, BTN_GREY, width=10).pack(side=tk.LEFT, padx=3)

        tk.Frame(top_row, bg=BORDER, width=1).pack(side=tk.LEFT, fill=tk.Y,
                                                    padx=(0, 20), pady=2)

        training_group = tk.Frame(top_row, bg=BG_CARD)
        training_group.pack(side=tk.LEFT, padx=(0, 20))
        tk.Label(training_group, text="Training", font=("Segoe UI", 8, "bold"),
                 fg=TEXT_LABEL, bg=BG_CARD).pack(anchor=tk.W, pady=(0, 4))
        training_btn_row = tk.Frame(training_group, bg=BG_CARD)
        training_btn_row.pack()
        self.train_q_button = self.make_button(training_btn_row, "Train Q-Learning",
                                         self.start_q_training, BTN_GREEN, width=16)
        self.train_q_button.pack(side=tk.LEFT, padx=3)
        self.train_dqn_button = self.make_button(training_btn_row, "Train DQN",
                                           self.start_dqn_training, BTN_VIOLET, width=12)
        self.train_dqn_button.pack(side=tk.LEFT, padx=3)
        self.view_q_table_button = self.make_button(training_btn_row, "View Q-Table",
                                             self.open_q_table_viewer, BTN_BLUE, width=14)
        self.view_q_table_button.pack(side=tk.LEFT, padx=3)

        self.status_label = tk.Label(top_row, text="Ready",
                                     font=("Segoe UI", 9, "italic"),
                                     fg=MUTED, bg=BG_CARD)
        self.status_label.pack(side=tk.RIGHT, padx=(16, 0))

        tk.Frame(controls, bg=BORDER, height=1).pack(fill=tk.X, padx=18)

        self.manual_panel = tk.Frame(controls, bg="#f8fafc",
                                     highlightthickness=2, highlightbackground=BORDER)
        self.manual_panel.pack(fill=tk.X, padx=16, pady=(12, 16))

        manual_header = tk.Frame(self.manual_panel, bg="#f8fafc")
        manual_header.pack(fill=tk.X, padx=14, pady=(12, 8))

        tk.Label(manual_header, text="Manual Policy Controls",
                 font=("Segoe UI", 9, "bold"), fg=TEXT_DARK, bg="#f8fafc"
                 ).pack(side=tk.LEFT)

        self.manual_status_label = tk.Label(
            manual_header,
            text="Select \"Manual\" policy above to enable action buttons",
            font=("Segoe UI", 9, "italic"), fg=TEXT_LABEL, bg="#f8fafc")
        self.manual_status_label.pack(side=tk.RIGHT)

        idle_group = tk.Frame(self.manual_panel, bg="#f8fafc")
        idle_group.pack(fill=tk.X, padx=14, pady=(0, 6))
        tk.Label(idle_group, text="Jammer IDLE", font=("Segoe UI", 8, "bold"),
                 fg=BTN_GREY, bg="#f8fafc", width=14, anchor=tk.W
                 ).pack(side=tk.LEFT, padx=(0, 8))
        idle_btn_row = tk.Frame(idle_group, bg="#f8fafc")
        idle_btn_row.pack(side=tk.LEFT, fill=tk.X, expand=True)

        active_group = tk.Frame(self.manual_panel, bg="#f8fafc")
        active_group.pack(fill=tk.X, padx=14, pady=(0, 12))
        tk.Label(active_group, text="Jammer ACTIVE", font=("Segoe UI", 8, "bold"),
                 fg=ACCENT_RED, bg="#f8fafc", width=14, anchor=tk.W
                 ).pack(side=tk.LEFT, padx=(0, 8))
        active_btn_row = tk.Frame(active_group, bg="#f8fafc")
        active_btn_row.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.manual_buttons = []
        self.manual_tooltips = []
        for index, spec in enumerate(MANUAL_ACTION_SPECS):
            parent_row = idle_btn_row if spec["group"] == "idle" else active_btn_row
            btn_text = f"{spec['label']}  [{spec['shortcut']}]"
            btn = tk.Button(
                parent_row, text=btn_text,
                command=lambda a=index: self.manual_action_trigger(a),
                font=("Segoe UI", 9, "bold"),
                relief="solid", bd=1, padx=14, pady=10,
                bg="#f8fafc", fg="#334155", cursor="arrow",
                activebackground=spec["color"],
                activeforeground="white",
                disabledforeground="#64748b",
                highlightbackground="#cbd5e1",
                state=tk.DISABLED)
            
            def on_manual_enter(event, button=btn, spec_color=spec["color"]):
                if button['state'] != 'disabled':
                    button.config(bg=self.lighten_color(spec_color, 0.1), highlightbackground=self.lighten_color(spec_color, 0.1))
            
            def on_manual_leave(event, button=btn, spec_color=spec["color"]):
                if button['state'] != 'disabled':
                    button.config(bg=spec_color, highlightbackground=spec_color)
            
            btn.bind("<Enter>", on_manual_enter)
            btn.bind("<Leave>", on_manual_leave)
            
            btn.pack(side=tk.LEFT, padx=4)
            tooltip = Tooltip(btn, spec["desc"])
            self.manual_buttons.append(btn)
            self.manual_tooltips.append(tooltip)

        self.root.bind("<Key>", self.on_manual_keypress)

        self.policy_mode.trace_add("write", lambda *args: self.on_policy_change())

        self.on_policy_change()

    def make_button(self, parent, text, command, color, width=12):
        btn = tk.Button(parent, text=text, command=command,
                         bg=color, fg="white",
                         font=("Segoe UI", 10, "bold"),
                         relief="solid", bd=1, cursor="hand2",
                         activebackground=self.lighten_color(color, 0.15), activeforeground="white",
                         disabledforeground=BTN_DISABLED,
                         highlightbackground=color,
                         padx=16, pady=8, width=width)
        
        def on_enter(event):
            btn.config(bg=self.lighten_color(color, 0.15), highlightbackground=self.lighten_color(color, 0.15))
        
        def on_leave(event):
            btn.config(bg=color, highlightbackground=color)
        
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        return btn

    def lighten_color(self, hex_color, factor):
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        lightened = tuple(min(int(c + (255 - c) * factor), 255) for c in rgb)
        return '#%02x%02x%02x' % lightened

    def get_policy_abbreviation(self):
        return self.policy_names.get(self.policy_mode.get(), "random")

    def get_action_block_reason(self, action_index: int) -> str:
        spec = MANUAL_ACTION_SPECS[action_index]
        jammer = self.environment.jammer_state
        data = self.environment.data_queue_level
        energy = self.environment.energy_queue_level
        if action_index == 0:
            return spec["desc"]
        if action_index == 1:
            if jammer == 1:
                return "Blocked: jammer is ACTIVE (Active-TX needs IDLE)"
            if data == 0:
                return "Blocked: data queue is empty"
            if energy < active_transmit_energy_per_packet:
                return f"Blocked: need ≥{active_transmit_energy_per_packet} energy (have {energy})"
            return spec["desc"]
        if action_index == 2:
            if jammer == 0:
                return "Blocked: jammer is IDLE (Harvest needs ACTIVE)"
            return spec["desc"]
        if action_index in (3, 4, 5, 6):
            if jammer == 0:
                return f"Blocked: jammer is IDLE ({spec['label']} needs ACTIVE)"
            if data == 0:
                return "Blocked: data queue is empty"
            return spec["desc"]
        return spec["desc"]

    def is_manual_mode_active(self) -> bool:
        return (self.get_policy_abbreviation() == "manual"
                and not self.running
                and not self.training_in_progress)

    def on_policy_change(self, event=None):
        if not hasattr(self, 'manual_buttons'):
            return
        is_manual = self.is_manual_mode_active()
        self.refresh_manual_buttons(is_manual)
        if is_manual:
            self.manual_panel.configure(bg="#eff6ff", highlightbackground=ACCENT_BLUE)
            for widget in self.manual_panel.winfo_children():
                try:
                    widget.configure(bg="#eff6ff")
                    for child in widget.winfo_children():
                        child.configure(bg="#eff6ff")
                        for grandchild in child.winfo_children():
                            grandchild.configure(bg="#eff6ff")
                except tk.TclError:
                    pass
            feasible = self.environment.get_possible_actions()
            names = ", ".join(get_action_name(a) for a in feasible)
            self.manual_status_label.configure(
                text=f"Manual mode — click a highlighted action or press 0–6  |  Available: {names}",
                fg=TEXT_DARK)
            if self._previous_policy != "manual":
                self.log_message("[Manual] Policy active — use action buttons or keys 0–6 to step.",
                                 tag="header")
        elif self.running:
            self.manual_panel.configure(bg="#f8fafc", highlightbackground=BORDER)
            self.manual_status_label.configure(
                text="Simulation running — pause to use manual controls", fg=ACCENT_ORANGE)
        elif self.training_in_progress:
            self.manual_panel.configure(bg="#f8fafc", highlightbackground=BORDER)
            self.manual_status_label.configure(
                text="Training in progress — manual controls locked", fg=ACCENT_VIOLET)
        else:
            self.manual_panel.configure(bg="#f8fafc", highlightbackground=BORDER)
            self.manual_status_label.configure(
                text="Select \"Manual\" policy above to enable action buttons", fg=TEXT_LABEL)
        self._previous_policy = self.get_policy_abbreviation()

    def refresh_manual_buttons(self, enable):
        possible_actions = self.environment.get_possible_actions() if enable else []
        for index, btn in enumerate(self.manual_buttons):
            spec = MANUAL_ACTION_SPECS[index]
            reason = self.get_action_block_reason(index)
            self.manual_tooltips[index].text = reason
            if enable and index in possible_actions:
                btn.configure(state=tk.NORMAL, bg=spec["color"], fg="white", cursor="hand2", relief="solid", bd=1, highlightbackground=spec["color"])
            elif enable:
                btn.configure(state=tk.DISABLED, bg="#e2e8f0", fg="#475569", cursor="not allowed", relief="solid", bd=1, highlightbackground="#cbd5e1")
            else:
                btn.configure(state=tk.DISABLED, bg="#f1f5f9", fg="#94a3b8", cursor="arrow", relief="solid", bd=1, highlightbackground="#cbd5e1")

    def on_manual_keypress(self, event):
        if not self.is_manual_mode_active():
            return
        key = event.char
        for index, spec in enumerate(MANUAL_ACTION_SPECS):
            if key == spec["shortcut"]:
                self.manual_action_trigger(index)
                return

    def toggle_run(self):
        if self.running:
            self.running = False
            self.play_button.configure(text="▶ Run", bg=BTN_BLUE)
            self.on_policy_change()
        else:
            if self.training_in_progress:
                messagebox.showwarning("Training in Progress",
                                     "Wait for training to finish first.")
                return
            self.running = True
            self.play_button.configure(text="⏸ Pause", bg=BTN_DANGER)
            self.refresh_manual_buttons(False)
            self.run_loop()

    def run_loop(self):
        if not self.running:
            return
        self.simulation_step()
        self.root.after(self.step_delay, self.run_loop)

    def simulation_step(self):
        if self.training_in_progress:
            return
        state = self.environment.get_state()
        possible_actions = self.environment.get_possible_actions()
        if not possible_actions:
            return

        mode = self.get_policy_abbreviation()
        if mode == "manual":
            self.log_message("[Step] Switch policy away from 'Manual' to use Step.",
                      tag="warn")
            return
        elif mode == "q" and self.q_table is not None:
            q_values = {action: self.q_table[state][action] for action in possible_actions}
            action = max(q_values, key=q_values.get)
        elif mode == "dqn" and self.dqn_model is not None:
            vector = np.zeros(total_states, dtype=np.float32)
            vector[state] = 1.0
            q_output = self.dqn_model(vector[np.newaxis], training=False)[0].numpy()
            mask = np.full(total_actions, -np.inf)
            for action in possible_actions:
                mask[action] = q_output[action]
            action = int(np.argmax(mask))
        else:
            action = int(np.random.choice(possible_actions))

        self.execute_action(action)

    def manual_action_trigger(self, action):
        if self.running:
            self.log_message("[Manual] Pause simulation before taking manual actions.", tag="warn")
            return
        if self.training_in_progress:
            self.log_message("[Manual] Wait for training to finish.", tag="warn")
            return
        if self.get_policy_abbreviation() != "manual":
            self.policy_mode.set("Manual")
        possible_actions = self.environment.get_possible_actions()
        if action not in possible_actions:
            reason = self.get_action_block_reason(action)
            self.log_message(f"[Manual] {get_action_name(action)} unavailable — {reason}",
                             tag="warn")
            return
        self.execute_action(action)

    def execute_action(self, action):
        pre_jammer = self.environment.jammer_state
        pre_data = self.environment.data_queue_level
        pre_energy = self.environment.energy_queue_level

        reward, next_state = self.environment.perform_action(action)

        self.last_action = action
        self.last_reward = reward
        self.total_packets += reward
        self.total_steps += 1

        self.jammer_label.configure(
            text=f"● {'ACTIVE' if self.environment.jammer_state == 1 else 'IDLE'}",
            fg=ACCENT_RED if self.environment.jammer_state == 1 else BTN_GREY)
        self.action_label.configure(text=get_action_name(action))
        self.reward_label.configure(
            text=f"Reward: +{reward} packets",
            fg=ACCENT_GREEN if reward > 0 else MUTED)

        average = self.total_packets / self.total_steps
        self.steps_label.configure(text=str(self.total_steps))
        self.packets_label.configure(text=str(self.total_packets))
        self.average_label.configure(text=f"{average:.4f} packets/slot")
        self.state_label.configure(
            text=f"(j={self.environment.jammer_state}, d={self.environment.data_queue_level}, e={self.environment.energy_queue_level})")

        self.draw_meter(self.data_meter, self.environment.data_queue_level, data_queue_capacity, ACCENT_BLUE)
        self.draw_meter(self.energy_meter, self.environment.energy_queue_level, energy_queue_capacity, ACCENT_GREEN)

        jammer_string = "ACTIVE" if pre_jammer == 1 else "IDLE"
        if reward > 0:
            tag = "reward"
        elif reward == 0 and action != 0:
            tag = "warn"
        else:
            tag = None
        self.log_message(f"[{self.total_steps:04d}] Jammer {jammer_string:6s} | "
                  f"{get_action_name(action):14s} | +{reward} pkts | "
                  f"d={self.environment.data_queue_level} e={self.environment.energy_queue_level}", tag=tag)

        self.draw_scene()
        self.schedule_animation(action, reward, pre_jammer)
        self.on_policy_change()

    def reset_simulation(self):
        self.running = False
        if self.animation_after_id:
            self.root.after_cancel(self.animation_after_id)
            self.animation_after_id = None
        self.play_button.configure(text="▶ Run", bg=BTN_BLUE)
        self.environment.reset()
        self.total_steps = 0
        self.total_packets = 0
        self.last_action = 0
        self.last_reward = 0
        self.jammer_label.configure(text="● IDLE", fg=JAMMER_IDLE_CLR)
        self.action_label.configure(text="—")
        self.reward_label.configure(text="Reward: —", fg=MUTED)
        self.steps_label.configure(text="0")
        self.packets_label.configure(text="0")
        self.average_label.configure(text="0.0000 packets/slot")
        self.state_label.configure(text="(j=0, d=0, e=0)")
        self.draw_meter(self.data_meter, 0, data_queue_capacity, ACCENT_BLUE)
        self.draw_meter(self.energy_meter, 0, energy_queue_capacity, ACCENT_GREEN)
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)
        self.log_message("System reset. Environment re-initialised.", tag="header")
        self.draw_scene()
        self.on_policy_change()

    def on_speed_change(self, value):
        self.step_delay = self.speed_variable.get()

    def lock_training(self, status_message, color):
        self.training_in_progress = True
        self.train_q_button.configure(state=tk.DISABLED)
        self.train_dqn_button.configure(state=tk.DISABLED)
        self.refresh_manual_buttons(False)
        self.status_label.configure(text=status_message, fg=color)

    def unlock_training(self):
        self.training_in_progress = False
        self.train_q_button.configure(state=tk.NORMAL)
        self.train_dqn_button.configure(state=tk.NORMAL)
        self.on_policy_change()

    def start_q_training(self):
        if self.training_in_progress or self.running:
            return
        self.lock_training("Q-Learning: Starting …", ACCENT_BLUE)
        threading.Thread(target=self.q_training_worker, daemon=True).start()

    def q_training_worker(self):
        try:
            from q_learning_agent import QLearningAgent
            import parameters

            steps = 200000
            agent = QLearningAgent(seed=42)
            state = agent.environment.reset()
            original_training_steps = parameters.q_learning_training_steps
            parameters.q_learning_training_steps = steps
            chunk_size = 5000

            for step in range(1, steps + 1):
                possible_actions = agent.environment.get_possible_actions()
                action = agent.select_action(state, possible_actions)
                reward, next_state = agent.environment.perform_action(action)
                next_possible_actions = agent.environment.get_possible_actions()
                agent.update_q_matrix(state, action, reward, next_state, next_possible_actions)
                agent.decay_exploration_rate()
                state = next_state
                if step % chunk_size == 0:
                    percentage = int(100 * step / steps)
                    self.root.after(0, self.status_label.configure,
                                    {"text": f"Q-Learning: {percentage}% ({step:,}/{steps:,})",
                                     "fg": ACCENT_BLUE})

            agent.save_q_table()
            self.q_table = agent.q_matrix
            parameters.q_learning_training_steps = original_training_steps
            self.root.after(0, self.on_training_done,
                            "Q-Learning complete — Q-Table saved.", True)
        except Exception as error:
            self.root.after(0, self.on_training_done,
                            f"Q-Learning failed: {error}", False)

    def start_dqn_training(self):
        if self.training_in_progress or self.running:
            return
        try:
            import tensorflow
        except ImportError:
            messagebox.showerror("Missing Dependency",
                                 "TensorFlow is required for DQN training.\n"
                                 "Run: pip install tensorflow")
            return
        self.lock_training("DQN: Initialising TensorFlow …", ACCENT_VIOLET)
        threading.Thread(target=self.dqn_training_worker, daemon=True).start()

    def dqn_training_worker(self):
        try:
            from deep_q_agent import DQNAgent
            import parameters

            steps = 5000
            agent = DQNAgent(seed=42)
            state_index = agent.environment.reset()
            state_vector = agent._one_hot_encode_state(state_index)
            original_training_steps = parameters.dqn_training_steps
            parameters.dqn_training_steps = steps
            chunk_size = 100

            for step in range(1, steps + 1):
                possible_actions = agent.environment.get_possible_actions()
                action = agent.select_action(state_vector, possible_actions)
                reward, next_state_index = agent.environment.perform_action(action)
                next_state_vector = agent._one_hot_encode_state(next_state_index)
                agent.replay_buffer.add_experience(state_vector, action, reward, next_state_vector, False)
                agent.train_step()
                agent.current_exploration_rate = max(parameters.minimum_exploration_rate,
                                                    agent.current_exploration_rate * parameters.exploration_decay_rate)
                if step % parameters.target_network_update_frequency == 0:
                    agent._synchronize_target_network()
                state_index, state_vector = next_state_index, next_state_vector
                if step % chunk_size == 0:
                    percentage = int(100 * step / steps)
                    self.root.after(0, self.status_label.configure,
                                    {"text": f"DQN: {percentage}% ({step}/{steps})",
                                     "fg": ACCENT_VIOLET})

            agent.save_model()
            self.dqn_model = agent.online_network
            parameters.dqn_training_steps = original_training_steps
            self.root.after(0, self.on_training_done,
                            "DQN training complete — model saved.", True)
        except Exception as error:
            self.root.after(0, self.on_training_done,
                            f"DQN training failed: {error}", False)

    def open_q_table_viewer(self):
        if self.q_table is None:
            messagebox.showinfo("Q-Table Not Available", 
                               "Q-table hasn't been trained yet. Please train Q-Learning first!")
            return
        
        q_window = tk.Toplevel(self.root)
        q_window.title("Q-Table Viewer")
        q_window.geometry("1400x800")
        q_window.configure(bg=BG_ROOT)
        
        info_frame = tk.Frame(q_window, bg=BG_CARD, padx=20, pady=16, highlightthickness=1, highlightbackground=BORDER)
        info_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(info_frame, text="Q-Table Viewer", font=("Segoe UI", 18, "bold"), fg=TEXT_DARK, bg=BG_CARD).pack(anchor=tk.W)
        tk.Label(info_frame, text=f"Loaded from: {self.q_table_path}", font=("Segoe UI", 10), fg=MUTED, bg=BG_CARD).pack(anchor=tk.W, pady=(4, 0))
        
        table_frame = tk.Frame(q_window, bg=BG_CARD, padx=20, pady=20, highlightthickness=1, highlightbackground=BORDER)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        columns = ("state_idx", "j", "d", "e", "Idle", "Active-TX", "Harvest", 
                   "Backscatter", "RA-0", "RA-1", "RA-2")
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=20)
        
        column_tooltips = {
            "state_idx": "State Index: Unique index representing the combined state (j, d, e)",
            "j": "Jammer State: 0 = Idle, 1 = Active\n\nDetermines which actions are feasible",
            "d": "Data Queue Level: Number of packets waiting to be transmitted\n\nCapacity: " + str(data_queue_capacity),
            "e": "Energy Queue Level: Amount of harvested energy available\n\nCapacity: " + str(energy_queue_capacity),
            "Idle": "Action 0: Idle\n\nDo nothing. Jammer state transitions naturally. No reward.",
            "Active-TX": "Action 1: Active Transmission\n\nTransmit packets when jammer is idle. Consumes energy.\n\nOnly feasible when j=0, d>0, e>0",
            "Harvest": "Action 2: Harvest Energy\n\nHarvest energy from jammer signal when active.\n\nOnly feasible when j=1",
            "Backscatter": "Action 3: Backscatter\n\nReflect jammer signal to transmit data when jammer is active.\n\nOnly feasible when j=1, d>0",
            "RA-0": "Action 4: Rate Adaptation Level 0\n\nTransmit using adaptive rate based on jammer power.\n\nOnly feasible when j=1, d>0",
            "RA-1": "Action 5: Rate Adaptation Level 1\n\nTransmit using adaptive rate based on jammer power.\n\nOnly feasible when j=1, d>0",
            "RA-2": "Action 6: Rate Adaptation Level 2\n\nTransmit using adaptive rate based on jammer power.\n\nOnly feasible when j=1, d>0",
        }
        
        for col in columns:
            tree.heading(col, text=col.replace("-", " "))
            tree.column(col, width=100, anchor=tk.CENTER)
        
        for state_index in range(len(self.q_table)):
            jammer, data, energy = decode_state(state_index)
            q_values = self.q_table[state_index]
            tree.insert("", tk.END, values=(
                state_index, jammer, data, energy,
                f"{q_values[0]:.4f}",
                f"{q_values[1]:.4f}",
                f"{q_values[2]:.4f}",
                f"{q_values[3]:.4f}",
                f"{q_values[4]:.4f}",
                f"{q_values[5]:.4f}",
                f"{q_values[6]:.4f}"
            ))
        
        style = ttk.Style(q_window)
        style.theme_use("clam")
        style.configure("Treeview", font=("Segoe UI", 10), rowheight=30,
                        background=BG_CARD, foreground=TEXT_DARK,
                        fieldbackground=BG_CARD, borderwidth=0)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"),
                        background=ACCENT_BLUE, foreground="white",
                        relief="flat")
        style.map("Treeview", background=[('selected', '#dbeafe')],
                  foreground=[('selected', TEXT_DARK)])
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        for col, tooltip_text in column_tooltips.items():
            for widget in table_frame.winfo_children():
                if isinstance(widget, ttk.Treeview):
                    for heading in widget.winfo_children():
                        pass
        
        header_tooltips = []
        def bind_header_tooltips():
            for col in columns:
                header_id = tree.heading(col)
                for widget in table_frame.winfo_children():
                    if isinstance(widget, ttk.Treeview):
                        pass
        
        formula_frame = tk.Frame(q_window, bg=BG_CARD, padx=20, pady=16, highlightthickness=1, highlightbackground=BORDER)
        formula_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(formula_frame, text="Q-Learning Formula", font=("Segoe UI", 14, "bold"), fg=TEXT_DARK, bg=BG_CARD).pack(anchor=tk.W)
        
        formula_text = """Bellman Equation (Q-Value Update):
Q(s, a) ← Q(s, a) + α [ r + γ maxₐ' Q(s', a') - Q(s, a) ]

Where:
- Q(s, a): Current Q-value for state s and action a
- α: Learning rate (controls how much new information overrides old)
- r: Reward received after taking action a in state s
- γ: Discount factor (determines importance of future rewards)
- maxₐ' Q(s', a'): Maximum Q-value for next state s' over all possible actions a'

State Encoding:
state_index = j × (data_queue_capacity + 1) × (energy_queue_capacity + 1) + d × (energy_queue_capacity + 1) + e

Where:
- j: Jammer state (0 or 1)
- d: Data queue level (0 to data_queue_capacity)
- e: Energy queue level (0 to energy_queue_capacity)
"""
        
        tk.Label(formula_frame, text=formula_text, font=("Consolas", 10),
                 fg=TEXT_DARK, bg=BG_CARD, justify=tk.LEFT).pack(anchor=tk.W, pady=(8, 0))

    def on_training_done(self, message, success):
        self.unlock_training()
        color = ACCENT_GREEN if success else ACCENT_RED
        self.status_label.configure(text=message, fg=color)
        tag = "train" if success else "warn"
        self.log_message(("✓ " if success else "✗ ") + message, tag=tag)
        if not success:
            messagebox.showerror("Training Error", message)

    def log_message(self, message, tag=None):
        self.log_text.configure(state=tk.NORMAL)
        if tag:
            self.log_text.insert(tk.END, message + "\n", tag)
        else:
            self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def draw_scene(self):
        self.canvas.delete("all")
        width = self.canvas.winfo_width() or 900
        height = self.canvas.winfo_height() or 500

        step = 32
        for grid_x in range(0, width, step):
            for grid_y in range(0, height, step):
                self.canvas.create_oval(grid_x-1, grid_y-1, grid_x+1, grid_y+1,
                                        fill="#e2e8f0", outline="")

        margin_x = width // 5
        self.transmitter_position = (margin_x, height - 130)
        self.receiver_position = (width - margin_x, height - 130)
        self.jammer_position = (width // 2, 110)

        self.draw_link(self.transmitter_position, self.receiver_position, "#cbd5e1", 3, solid=True)
        self.draw_link(self.jammer_position, self.transmitter_position, "#fecdd3", 2, solid=False)
        self.draw_link(self.jammer_position, self.receiver_position, "#fecdd3", 2, solid=False)

        mid_x = (self.transmitter_position[0] + self.receiver_position[0]) // 2
        mid_y = (self.transmitter_position[1] + self.receiver_position[1]) // 2
        self.canvas.create_text(mid_x, mid_y - 18, text="DATA CHANNEL",
                                 font=("Segoe UI", 9, "bold"),
                                 fill="#64748b")

        self.draw_jammer()
        self.draw_transmitter()
        self.draw_receiver()

        self.draw_inline_queues()

    def draw_link(self, start, end, color, line_width, solid=True):
        dash = None if solid else (8, 5)
        self.canvas.create_line(start[0], start[1], end[0], end[1],
                                fill=color, width=line_width,
                                dash=dash)

    def draw_jammer(self):
        center_x, center_y = self.jammer_position
        active = self.environment.jammer_state == 1
        rim_color = ACCENT_RED if active else JAMMER_IDLE_CLR
        glow_color = "#fee2e2" if active else BG_CANVAS
        label_text = "JAMMER ● ACTIVE" if active else "JAMMER ○ IDLE"

        halo_radius = 52
        self.canvas.create_oval(center_x - halo_radius, center_y - halo_radius,
                                center_x + halo_radius, center_y + halo_radius,
                                fill=glow_color, outline=rim_color, width=1 if active else 0)

        radius = 34
        points = []
        import math
        for i in range(6):
            angle = math.radians(60 * i - 30)
            points += [center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)]
        self.canvas.create_polygon(*points, fill="#f8fafc" if not active else "#fff1f2",
                                   outline=rim_color, width=3)

        self.canvas.create_text(center_x, center_y, text="⚡", font=("Segoe UI", 18),
                                 fill=rim_color)

        if active:
            for wave_radius in [60, 78, 96]:
                self.canvas.create_oval(center_x - wave_radius, center_y - wave_radius,
                                        center_x + wave_radius, center_y + wave_radius,
                                        outline=rim_color, width=1,
                                        dash=(5, 7))

        self.canvas.create_text(center_x, center_y + 60, text=label_text,
                                 font=("Segoe UI", 10, "bold"),
                                 fill=rim_color)

    def draw_transmitter(self):
        center_x, center_y = self.transmitter_position
        self.canvas.create_oval(center_x - 44, center_y - 44, center_x + 44, center_y + 44,
                                fill="#dbeafe", outline=ACCENT_BLUE, width=2)
        self.canvas.create_line(center_x, center_y - 32, center_x, center_y + 24,
                                fill=ACCENT_BLUE, width=4)
        self.canvas.create_line(center_x - 16, center_y - 12, center_x + 16, center_y - 12,
                                fill=ACCENT_BLUE, width=3)
        self.canvas.create_line(center_x - 10, center_y + 6, center_x + 10, center_y + 6,
                                fill=ACCENT_BLUE, width=3)
        self.canvas.create_oval(center_x - 6, center_y - 38, center_x + 6, center_y - 26,
                                fill=ACCENT_BLUE, outline="white", width=2)
        self.canvas.create_rectangle(center_x - 14, center_y + 22, center_x + 14, center_y + 30,
                                     fill=ACCENT_BLUE, outline="")
        self.canvas.create_text(center_x, center_y + 52, text="TRANSMITTER (TX)",
                                 font=("Segoe UI", 10, "bold"),
                                 fill=ACCENT_BLUE)

    def draw_receiver(self):
        center_x, center_y = self.receiver_position
        self.canvas.create_oval(center_x - 44, center_y - 44, center_x + 44, center_y + 44,
                                fill="#d1fae5", outline=ACCENT_GREEN, width=2)
        self.canvas.create_arc(center_x - 34, center_y - 34, center_x + 14, center_y + 34,
                               start=300, extent=120,
                               style=tk.ARC, outline=ACCENT_GREEN, width=4)
        self.canvas.create_line(center_x - 8, center_y, center_x + 20, center_y - 14,
                                fill=ACCENT_GREEN, width=3)
        self.canvas.create_oval(center_x + 17, center_y - 18, center_x + 25, center_y - 10,
                                fill=ACCENT_GREEN, outline="white", width=2)
        self.canvas.create_line(center_x - 8, center_y, center_x - 8, center_y + 26,
                                fill=ACCENT_GREEN, width=3)
        self.canvas.create_rectangle(center_x - 20, center_y + 24, center_x + 4, center_y + 30,
                                     fill=ACCENT_GREEN, outline="")
        self.canvas.create_text(center_x, center_y + 52, text="RECEIVER (RX)",
                                 font=("Segoe UI", 10, "bold"),
                                 fill=ACCENT_GREEN)

    def draw_inline_queues(self):
        transmitter_x, transmitter_y = self.transmitter_position
        base_y = transmitter_y + 72

        segment_width, segment_height, gap = 13, 15, 3

        for (label, value, capacity, color) in [
            ("DATA", self.environment.data_queue_level, data_queue_capacity, ACCENT_BLUE),
            ("ENERGY", self.environment.energy_queue_level, energy_queue_capacity, ACCENT_GREEN),
        ]:
            self.canvas.create_text(transmitter_x - 6, base_y + segment_height // 2,
                                     text=label, font=("Segoe UI", 8, "bold"),
                                     fill=color, anchor=tk.E)
            x0 = transmitter_x
            for i in range(capacity):
                fill = color if i < value else "#e2e8f0"
                self.canvas.create_rectangle(
                    x0 + i * (segment_width + gap), base_y,
                    x0 + i * (segment_width + gap) + segment_width, base_y + segment_height,
                    fill=fill, outline="#cbd5e1", width=0
                )
            self.canvas.create_text(
                x0 + capacity * (segment_width + gap) + 6, base_y + segment_height // 2,
                text=f"{value}/{capacity}", font=("Segoe UI", 8),
                fill=MUTED, anchor=tk.W
            )
            base_y += segment_height + 10

    def schedule_animation(self, action, reward, pre_jammer):
        transmitter = self.transmitter_position
        receiver = self.receiver_position
        jammer = self.jammer_position

        if action == 1 and pre_jammer == 0:
            self.animate_pulse(transmitter, receiver, ACCENT_BLUE, reward, tag="pulse1")

        elif action == 2 and pre_jammer == 1:
            self.animate_pulse(jammer, transmitter, ACCENT_GREEN, 0, label="+ ENERGY",
                                tag="pulse1")

        elif action == 3 and pre_jammer == 1:
            self.animate_pulse(jammer, transmitter, ACCENT_ORANGE, 0, label="← REFLECT",
                                tag="pulse1",
                                done_callback=lambda: self.animate_pulse(
                                    transmitter, receiver, ACCENT_AMBER, reward, tag="pulse2"))

        elif 4 <= action <= 6 and pre_jammer == 1:
            self.animate_pulse(jammer, receiver, ACCENT_RED, 0, label="✕ JAM", tag="pulse1")
            self.animate_pulse(transmitter, receiver, ACCENT_ORANGE, reward, tag="pulse2")

    def animate_pulse(self, start, end, color, reward,
                       label=None, tag="pulse", step=0,
                       total_steps=18, done_callback=None):
        if step > total_steps:
            self.canvas.delete(tag)
            if done_callback:
                done_callback()
            return

        ratio = step / total_steps
        x = start[0] + ratio * (end[0] - start[0])
        y = start[1] + ratio * (end[1] - start[1])

        self.canvas.delete(tag)
        radius = 10 if reward > 0 else 6
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                fill=color, outline="white", width=2,
                                tags=tag)
        if label:
            self.canvas.create_text(x, y - 20, text=label,
                                     font=("Segoe UI", 8, "bold"),
                                     fill=color, tags=tag)
        elif reward > 0:
            self.canvas.create_text(x, y - 20,
                                     text=f"+{reward} pkts",
                                     font=("Segoe UI", 8, "bold"),
                                     fill=ACCENT_GREEN, tags=tag)

        self.animation_after_id = self.root.after(
            24,
            lambda: self.animate_pulse(start, end, color, reward,
                                        label, tag, step + 1,
                                        total_steps, done_callback))


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.25)
    except Exception:
        pass

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TCombobox",
                    fieldbackground=BG_CARD,
                    background=BG_CARD,
                    foreground=TEXT_DARK,
                    selectbackground=ACCENT_BLUE,
                    selectforeground="white",
                    borderwidth=1,
                    relief="flat")

    app = AmbientJammingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

