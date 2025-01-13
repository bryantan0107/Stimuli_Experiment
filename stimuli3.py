from psychopy import visual, sound, core, event
from psychopy.parallel import ParallelPort
import numpy as np
import random
import sounddevice as sd
import sys
import threading


# Parameters
class ExperimentConfig:
    SAMPLE_RATE = 48000
    DURATION = 2
    LEFT_CARRIER_FREQ = 440
    RIGHT_CARRIER_FREQ = 340
    MOD_FREQ_HIGH = 40
    MOD_FREQ_LOW = 25
    VISUAL_FREQ_HIGH = 18
    VISUAL_FREQ_LOW = 13
    NUM_BLOCKS = 6
    TRIALS_PER_BLOCK = 40  # should be 40
    BREAK_TIME = 60  # 休息时间（秒）
    SF = 0.035
    REFRESH_RATE = 60
    REST_DURATION = 3

# Function to create modulated tones


class AudioManager:
    @staticmethod
    def create_modulated_tone(mod_freq, carrier_freq):
        time = np.linspace(0, ExperimentConfig.DURATION, int(
            ExperimentConfig.SAMPLE_RATE * ExperimentConfig.DURATION), endpoint=False)
        am_wave = (1 + np.sin(2 * np.pi * mod_freq * time)) * \
            np.sin(2 * np.pi * carrier_freq * time)
        am_wave /= np.max(np.abs(am_wave))  # Normalize wave amplitude
        return am_wave.astype(np.float32), mod_freq


# Create tones
tone_low, tone_low_freq = AudioManager.create_modulated_tone(
    ExperimentConfig.MOD_FREQ_LOW, ExperimentConfig.RIGHT_CARRIER_FREQ)
tone_high, tone_high_freq = AudioManager.create_modulated_tone(
    ExperimentConfig.MOD_FREQ_HIGH, ExperimentConfig.LEFT_CARRIER_FREQ)

# Define stereo audio combinations
stereo_combinations = {
    "right_high_left_low": (tone_high, tone_low),
    "right_low_left_high": (tone_low, tone_high),
}


# Initialize PsychoPy window
win = visual.Window(
    size=(1440, 900),  # Full HD resolution
    color=(-1, -1, -1),  # Background color (black)
    units="pix",
    fullscr=True,  # Enable full screen
    allowGUI=False,
    screen=1
)


def check_escape(win):
    keys = event.getKeys(keyList=["escape"])
    if "escape" in keys:
        win.close()
        core.quit()

# Trigger sender class


class BittiumTriggerSender:
    def __init__(self, parallel_port: ParallelPort, trigger_duration: float) -> None:
        self.parallel_port = parallel_port
        self.trigger_duration = trigger_duration

    def send_trigger(self, trigger_code: int) -> None:
        if sys.platform == "darwin":  # macOS
            print(f"Trigger sent: {trigger_code}")
        else:
            try:
                print(f"Trigger sent: {trigger_code}")
                # Send the trigger (setData sets the port pins to the value of trigger)
                clock = core.Clock()
                self.parallel_port.setData(trigger_code)
                # Keep the trigger high for the specified time
                while clock.getTime() < self.trigger_duration:
                    pass
                self.parallel_port.setData(0)  # Reset trigger to low
                # Small pause after resetting the trigger
                while clock.getTime() < self.trigger_duration:
                    pass
            except Exception as e:
                print(f"Error sending trigger: {e}")

# Function to check experimental setup


def show_pre_experiment_check(win):
    """
    Display a pre-experiment checklist to ensure setup is ready.
    :param win: PsychoPy window instance
    """
    checklist_text = visual.TextStim(
        win,
        text=("Hello, before we start, please let the experimenter to confirm the following:\n\n"
              "1. The keyboard is positioned correctly and is ready to use.\n"
              "2. The audio volume is set to a comfortable level.\n\n"
              "If everything is ready, press any key to proceed."),
        color="white",
        pos=(0, 0),
        wrapWidth=1200,
    )
    checklist_text.draw()
    win.flip()
    event.waitKeys()  # Wait for participant to confirm

# Function to display introduction text and wait for a key press


def show_introduction(win):
    intro_text = visual.TextStim(
        win,
        text="Welcome.\n\n"
             "You will now go through a resting state phase before the experiment start.\n\n"
             "There's nothing you need to do beside open or close your eyes.\n\n"
             "Please relax and focus.\n\n"
             "Press any key to begin.",
        color="white", pos=(0, 0), wrapWidth=1200,
    )
    intro_text.draw()
    win.flip()
    event.waitKeys()  # Wait for any key press


def countdown(win, rest_text):
    countdown_text = visual.TextStim(
        win,
        color="white",
        pos=(0, -100),
    )

    countdown_time = 4
    start_time = core.getTime()
    while core.getTime() - start_time < countdown_time:
        elapsed_time = core.getTime() - start_time
        remaining_time = countdown_time - elapsed_time

        # Update progress bar width
        countdown_text.text = f"{int(remaining_time)}"

        # Draw performance text, progress bar, and countdown timer
        rest_text.draw()
        countdown_text.draw()
        win.flip()

# RestingState class: manages the resting state display and trigger sending


class RestingState:
    def __init__(self, win, trigger_sender, duration=3, eyes_open=True):
        self.win = win
        self.trigger_sender = trigger_sender
        self.duration = duration
        self.eyes_open = eyes_open
        self.state = "Eyes Open" if eyes_open else "Eyes Closed"
        self.fixation = visual.TextStim(
            self.win, text="+", color="white", height=70  # Fixation cross
        )
        self.rest_text = visual.TextStim(
            self.win, text="", color="white", pos=(0, 0))

    def show_fixation(self):
        """Show the fixation cross"""
        clock = core.Clock()
        clock.reset()  # Reset the clock to 0

        while clock.getTime() < self.duration:
            self.fixation.draw()
            self.win.flip()
            check_escape(self.win)

    def update_rest_text(self):
        """Update resting state text based on eye state"""
        if self.state == "Eyes Open":
            self.rest_text.text = "You will see a cross in few seconds,\n\n please look at the cross and keep your eyes open."
        else:
            self.rest_text.text = "Now close your eyes.\n\nOpen your eyes again when you hear a beep sound."

    def start(self):
        """Start the resting state and send triggers"""
        self.update_rest_text()
        self.rest_text.draw()
        win.flip()
        core.wait(3)

        countdown(win, self.rest_text)

        self.trigger_sender.send_trigger(
            2 if self.eyes_open else 3)  # RS_start_EO or RS_start_EC

        self.show_fixation()  # Show fixation cross

        self.trigger_sender.send_trigger(
            4 if self.eyes_open else 5)  # RE_start_EO or RE_start_EC

        # Play a sound only if the eyes were closed
        if not self.eyes_open:  # Only play sound if eyes were closed
            # Parameters for the beep sound
            SAMPLE_RATE = 48000  # Sample rate for audio
            DURATION = 0.5  # Duration of the beep sound in seconds
            FREQUENCY = 1440  # Frequency of the beep sound in Hz (A4 note)

            # Generate the beep sound
            time = np.linspace(0, DURATION, int(
                SAMPLE_RATE * DURATION), endpoint=False)
            beep = np.sin(2 * np.pi * FREQUENCY * time)

            # Play the beep sound
            sd.play(beep, samplerate=SAMPLE_RATE)
            sd.wait()  # Wait until the sound is finished

        # while clock.getTime() - start_time < self.duration:
        #     keys = event.getKeys(keyList=["escape"])
        #     if "escape" in keys:
        #         self.win.close()
        #         core.quit()

    def set_duration(self, duration):
        """Optionally change the duration of the resting state"""
        self.duration = duration

    def set_eyes_open(self, eyes_open):
        """Optionally change whether the eyes are open or closed"""
        self.eyes_open = eyes_open
        self.state = "Eyes Open" if eyes_open else "Eyes Closed"


# Inter-block break with progress bar and performance display
def show_break_with_progress(win, break_time, correct_answers, total_trials):
    # Define the progress bar and its outline
    progress_text = visual.TextStim(
        win,
        text="Please take a break.\n\nTime remaining: 60 seconds",
        color="white",
        pos=(0, 0),
    )
    score_text = visual.TextStim(
        win,
        text=f"Here is your result from the last block:\n {correct_answers}/{total_trials} correct answers",
        color="white",
        pos=(0, 70),
    )

    start_time = core.getTime()
    while core.getTime() - start_time < break_time:
        elapsed_time = core.getTime() - start_time
        remaining_time = break_time - elapsed_time

        # Update progress bar width
        progress_text.text = f"Please take a break.\n\nTime remaining: {int(remaining_time)} seconds"

        # Draw performance text, progress bar, and countdown timer
        score_text.draw()
        progress_text.draw()
        win.flip()


class Stimulus:
    def __init__(self, win, duration):
        """
        Initialize the base Stimulus class.
        :param win: PsychoPy window instance
        :param duration: Duration of the stimulus in seconds
        """
        self.win = win
        self.duration = duration

    def present(self):
        """Display the stimulus (to be implemented in subclasses)."""
        raise NotImplementedError("Subclasses must implement this method.")


class VisualStimulus(Stimulus):
    def __init__(self, win, trigger_sender, left_freq, right_freq, sf=ExperimentConfig.SF, size=300):
        """
        Initialize the visual stimulus.
        :param win: PsychoPy window instance
        :param left_freq: Frequency for the left visual stimulus
        :param right_freq: Frequency for the right visual stimulus
        :param sf: Spatial frequency
        :param size: Size of the visual stimulus
        """
        super().__init__(win, duration=None)  # Visual stimuli are not duration-specific
        self.left_grating = visual.GratingStim(
            win, tex="sin", mask="gauss", sf=sf, units="pix", size=(350, 350), pos=(-200, 0), ori=0)
        self.right_grating = visual.GratingStim(
            win, tex="sin", mask="gauss", sf=sf, units="pix", size=(350, 350), pos=(200, 0), ori=0)
        self.left_freq = left_freq
        self.right_freq = right_freq
        self.trigger_sender = trigger_sender
        # Create visual stimuli
        # self.left_grating = visual.Rect(
        #     win, width=200, height=200, pos=(-200, 0), fillColor="white"
        # )
        # self.right_grating = visual.Rect(
        #     win, width=200, height=200, pos=(200, 0), fillColor="white"
        # )

    def present(self, duration=None, response_handler=None, block_trial_text=None):
        """Present the visual stimulus for the specified duration."""
        self.trigger_sender.send_trigger(
            21 if self.left_freq == ExperimentConfig.VISUAL_FREQ_HIGH else 22)

        # REFRESH_RATE = win.getActualFrameRate() or 60
        # # Frames per cycle calculation for flickering
        # frames_per_cycle_left = int(REFRESH_RATE / self.left_freq)
        # frames_per_cycle_right = int(REFRESH_RATE / self.right_freq)
        # on_frames_left = frames_per_cycle_left // 2
        # off_frames_left = frames_per_cycle_left - on_frames_left
        # on_frames_right = frames_per_cycle_right // 2
        # off_frames_right = frames_per_cycle_right - on_frames_right

        # # Initialize frame counters and states
        # frame_n_left = 0
        # frame_n_right = 0
        # left_on = True
        # right_on = True
        # last_left_frame = frame_n_left
        # last_right_frame = frame_n_right

        # Run the visual training loop
        clock = core.Clock()
        start_time = clock.getTime()
        while duration is None or clock.getTime() - start_time < duration:
            current_time = clock.getTime() - start_time  # 获取当前相对时间

            # user_response = None

            keys = event.getKeys(keyList=["left", "right", "escape"])
            if keys:
                user_response = keys[0]  # 保存用户响应
                if "escape" in keys:  # 如果按下 ESC，直接退出程序
                    self.win.close()
                    core.quit()
                break  # 捕获按键后中断

            if response_handler and response_handler():  # 外部传入的逻辑生效
                break

            # 计算左侧和右侧刺激的正弦波亮度值
            left_brightness = 0.5 + 0.5 * \
                np.sin(2 * np.pi * self.left_freq * current_time)
            right_brightness = 0.5 + 0.5 * \
                np.sin(2 * np.pi * self.right_freq * current_time)

            # 应用调整后的对比度
            self.left_grating.contrast = left_brightness
            self.right_grating.contrast = right_brightness

            # # Update left grating state
            # if left_on and frame_n_left >= last_left_frame + on_frames_left:
            #     left_on = False
            #     last_left_frame = frame_n_left
            # if not left_on and frame_n_left >= last_left_frame + off_frames_left:
            #     left_on = True
            #     last_left_frame = frame_n_left

            # # Update right grating state
            # if right_on and frame_n_right >= last_right_frame + on_frames_right:
            #     right_on = False
            #     last_right_frame = frame_n_right
            # if not right_on and frame_n_right >= last_right_frame + off_frames_right:
            #     right_on = True
            #     last_right_frame = frame_n_right

            # frame_n_left += 1
            # frame_n_right += 1

            # # 应用对比度条件
            # if self.left_freq == ExperimentConfig.VISUAL_FREQ_LOW:
            #     self.left_grating.contrast = 0.5

            # if self.right_freq == ExperimentConfig.VISUAL_FREQ_LOW:
            #     self.right_grating.contrast = 0.5

            # # Draw gratings based on current states
            # if left_on:
            #     self.left_grating.draw()
            # if right_on:
            #     self.right_grating.draw()

            # 绘制刺激
            self.left_grating.draw()
            self.right_grating.draw()

            win.flip()  # Update the window

            # Optionally draw block_trial_text
            if block_trial_text is not None:
                block_trial_text.draw()

        self.trigger_sender.send_trigger(
            25 if self.left_freq == ExperimentConfig.VISUAL_FREQ_HIGH else 26)

        return None if duration is not None else user_response


class AuditoryStimulus(Stimulus):
    def __init__(self, trigger_sender, left_tone, right_tone, left_tone_freq):
        """
        Initialize the auditory stimulus.
        :param left_tone: Audio waveform for the left ear
        :param right_tone: Audio waveform for the right ear
        """
        super().__init__(win=None, duration=None)  # Auditory stimuli do not require a window
        self.stereo_wave = np.column_stack(
            (left_tone, right_tone)).astype(np.float32)
        self.left_tone_freq = left_tone_freq
        self.trigger_sender = trigger_sender

    def present(self, duration=None, response_handler=None):
        """Play the auditory stimulus for the specified duration."""
        self.trigger_sender.send_trigger(
            23 if self.left_tone_freq == ExperimentConfig.MOD_FREQ_HIGH else 24)

        stream = sd.OutputStream(
            samplerate=ExperimentConfig.SAMPLE_RATE, channels=2)

        try:
            stream.start()

            clock = core.Clock()
            audio_index = 0
            buffer_size = 1024

            while True:
                # 停止条件：有duration时按时间停止；否则按response_handler停止
                if duration is not None and clock.getTime() >= duration:
                    break
                elif response_handler and response_handler():
                    break

                # 动态写入音频数据
                if audio_index < len(self.stereo_wave):
                    stream.write(
                        self.stereo_wave[audio_index:audio_index + buffer_size])
                    audio_index += buffer_size
                else:
                    audio_index = 0  # 循环播放音频
        finally:
            # 确保关闭音频流
            stream.stop()
            stream.close()

        self.trigger_sender.send_trigger(
            27 if self.left_tone_freq == ExperimentConfig.MOD_FREQ_HIGH else 28)


# Visual Training with flickering
class TrainingManager:
    def __init__(self, win, trigger_sender):
        self.win = win
        self.trigger_sender = trigger_sender

    def run_visual_training(self, left_freq, right_freq, duration, instruction_text):
        # Show instruction
        training_text = visual.TextStim(
            self.win, text=instruction_text, color="white")
        training_text.draw()
        self.win.flip()
        core.wait(2)

        countdown(win, training_text)

        visual_stimulus = VisualStimulus(
            win, self.trigger_sender, left_freq, right_freq)
        visual_stimulus.present(duration)

    def run_auditory_training(self, left_tone, right_tone, left_tone_freq, duration, instruction_text):
        """
        Runs auditory training with specified left and right tones for a given duration.
        :param left_tone: The audio waveform for the left ear
        :param right_tone: The audio waveform for the right ear
        :param duration: Duration of the training in seconds
        :param instruction_text: Instruction to display before training starts
        """
        # Show instruction
        training_text = visual.TextStim(
            self.win, text=instruction_text, color="white")
        training_text.draw()
        self.win.flip()
        core.wait(3)

        countdown(win, training_text)

        auditory_stimulus = AuditoryStimulus(
            self.trigger_sender, left_tone, right_tone, left_tone_freq)
        auditory_stimulus.present(duration)


class Experiment:
    def __init__(self, win, config, trigger_sender, stimuli_combinations, stereo_combinations, block_scores):
        """
        Initialize the Experiment class.
        :param win: PsychoPy window instance
        :param config: Experiment configuration object
        :param trigger_sender: Instance of TriggerManager to handle triggers
        :param stimuli_combinations: List of visual and auditory stimulus combinations
        :param stereo_combinations: Dictionary of stereo audio combinations
        """
        self.win = win
        self.config = config
        self.trigger_sender = trigger_sender
        self.stimuli_combinations = stimuli_combinations
        self.stereo_combinations = stereo_combinations
        self.block_scores = block_scores

    def run_practice_session(self, focus_type, trials):
        """
        Run a practice session based on the focus type.
        :param focus_type: The type of focus for this practice session
        """
        practice_instruction_text = {
            "Visual first, then Auditory": "This is a practice session where you can practice to focus on the visual stimuli first.\n\n"
                                           "Press any key to start the practice session.",
            "Auditory first, then Visual": "This is a practice session where you can practice to focus on the auditory stimuli first.\n\n"
                                           "Press any key to start the practice session."
        }

        # Display instructions for the practice session
        instruction = visual.TextStim(
            self.win,
            text=practice_instruction_text[focus_type],
            color="white",
            pos=(0, 0)
        )
        instruction.draw()
        self.win.flip()
        event.waitKeys()  # Wait for user to acknowledge

        # Randomly select a practice trial
        practice_trial = random.choice(trials)

        # Create a single practice trial
        # Example trial configuration
        self.run_trial(-1, practice_trial, 0)

        # Show feedback and ask if the user is ready to proceed
        while True:
            feedback_text = "Did you feel ready completing the task?\n\n" \
                            "Press 'right' to proceed to the trial.\n" \
                            "Press 'left' to repeat the practice session."
            feedback = visual.TextStim(
                self.win, text=feedback_text, color="white", pos=(0, 0)
            )
            feedback.draw()
            self.win.flip()

            keys = event.waitKeys(keyList=["left", "right"])
            if "right" in keys:
                break  # Exit the practice session
            elif "left" in keys:
                # Repeat the practice session
                self.run_trial(-1, practice_trial, 0)

    def show_block_focus(self, block, focus_type):
        """
        Display instructions for a specific block based on the focus type.
        :param block: Current block number
        :param focus_type: The type of focus for this block
        """
        focus_messages = {
            "None": "There's no instruction in this block. \n\n",
            "Visual first, then Auditory": "In this block, please focus on the visual stimuli first.\n\n",
            "Auditory first, then Visual": "In this block, please focus on the auditory stimuli first.\n\n",
        }
        instruction_text = f"Block {block}/{ExperimentConfig.NUM_BLOCKS} Instruction:\n\n" + \
            focus_messages[focus_type]
        instruction_text += "Press any key to start this block."

        # Display the instruction text
        block_instruction = visual.TextStim(
            win, text=instruction_text, color="white", pos=(0, 0))
        block_instruction.draw()
        win.flip()
        event.waitKeys()  # Wait for participant to acknowledge

    def show_rule_reminder(self, win, focus_type):
        """
        Display the block-specific focus instructions along with the general experiment rules.
        :param win: PsychoPy window instance
        :param block: Current block number
        :param focus_type: The type of focus for this block
        """
        # Define block-specific focus messages
        focus_messages = {
            "None": "There's no special focus in this block.\n\n",
            "Visual first, then Auditory": "In this block, please focus on the visual stimuli first.\n\n",
            "Auditory first, then Visual": "In this block, please focus on the auditory stimuli first.\n\n",
        }

        # Combine block-specific focus with general rules
        reminder_text = (
            "Due to the large number of blocks and trials in this experiment, we want to ensure you don’t forget your task.\n\n "
            "Here is a reminder to help you stay on track:\n\n"
            "Your task is to identify which visual and auditory stimuli have the higher frequency.\n\n"
            "If the higher frequency stimuli of visual and auditory both are on the right side, press the 'right' key.\n"
            "Otherwise, press the 'left' key.\n\n"
            f"Here are the instructions for this block:\n\n"
            + focus_messages.get(focus_type, "")
            + "Press any key to continue the experiment."
        )

        # Display the combined instruction and rule reminder
        reminder_stim = visual.TextStim(
            win,
            text=reminder_text,
            color="white",
            pos=(0, 0),
            wrapWidth=1200
        )
        reminder_stim.draw()
        self.win.flip()
        event.waitKeys()

    def present_visual_and_auditory(self, visual_stimulus, auditory_stimulus, block_trial_text):
        """
        同时呈现视觉和听觉刺激，并返回用户的响应。
        :param visual_stimulus: 视觉刺激对象
        :param auditory_stimulus: 听觉刺激对象
        :param response_handler: 用户响应处理函数
        :return: 用户响应
        """
        stop_flag = threading.Event()
        audio_finished_flag = threading.Event()

        # 定义 response_handler
        def response_handler():
            return stop_flag.is_set()  # 子线程依赖此函数检测是否需要停止

        def play_audio():
            """音频播放线程"""
            try:
                auditory_stimulus.present(
                    duration=None, response_handler=response_handler)
            except Exception as e:
                print(f"音频播放时出错: {e}")
            finally:
                audio_finished_flag.set()

        # 启动音频播放线程
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()

        # 主线程处理视觉刺激和按键检测
        user_response = None

        try:
            user_response = visual_stimulus.present(
                duration=None,
                response_handler=lambda: stop_flag.is_set() or audio_finished_flag.is_set(),
                block_trial_text=block_trial_text
            )

        finally:
            stop_flag.set()
            audio_thread.join()  # 确保音频线程正常结束

        return user_response

    def run_trial(self, trial_index, trial, block, focus_type=None):
        """
        Execute a single trial.
        :param trial_index: The index of the trial within the block
        :param trial: The trial's stimulus configuration
        :param block: The current block number
        :return: User response, visual_high position, and audio_high position
        """

        # Extract visual and auditory conditions
        visual_high = trial["visual_high"]
        audio_high = trial["audio_high"]

        # Display trial information
        block_trial_text = visual.TextStim(
            self.win,
            text=f"Block: {block}/{self.config.NUM_BLOCKS}, Trial: {trial_index + 1}/{self.config.TRIALS_PER_BLOCK}",
            color="white",
            pos=(0, 300),
        )

        # Send trial start trigger
        self.trigger_sender.send_trigger(20)

        visual_stimulus = VisualStimulus(
            self.win,
            self.trigger_sender,
            left_freq=ExperimentConfig.VISUAL_FREQ_HIGH if visual_high == "left" else ExperimentConfig.VISUAL_FREQ_LOW,
            right_freq=ExperimentConfig.VISUAL_FREQ_HIGH if visual_high == "right" else ExperimentConfig.VISUAL_FREQ_LOW
        )

        auditory_stimulus = AuditoryStimulus(
            self.trigger_sender,
            left_tone=tone_high if audio_high == "left" else tone_low,
            right_tone=tone_high if audio_high == "right" else tone_low,
            left_tone_freq=tone_high_freq if audio_high == "left" else tone_low_freq
        )

        # Present both stimuli and capture response
        response = self.present_visual_and_auditory(
            visual_stimulus, auditory_stimulus, block_trial_text)
        trigger_sender.send_trigger(30)  # trial_end
        win.flip()
        core.wait(0.5)

        if focus_type and (trial_index + 1) % 10 == 0 and (trial_index + 1) != self.config.TRIALS_PER_BLOCK:
            self.show_rule_reminder(self.win, focus_type)

        return response, visual_high, audio_high

    def run_block(self, block, focus_type):
        """
        Run a single block of the experiment.
        :param block: The current block number
        :param focus_type: The type of focus for this block
        :return: Number of correct answers in this block
        """
        trigger_sender.send_trigger(19)  # block_start

        self.show_block_focus(block, focus_type)

        correct_answers = 0
        trials = self.stimuli_combinations * \
            (self.config.TRIALS_PER_BLOCK // len(self.stimuli_combinations))
        random.shuffle(trials)

        # Add practice session for specific blocks
        if focus_type in ["Visual first, then Auditory", "Auditory first, then Visual"]:
            self.run_practice_session(focus_type, trials)

        for trial_index, trial in enumerate(trials):
            response, visual_high, audio_high = self.run_trial(
                trial_index, trial, block, focus_type)

            correct_response = "right" if visual_high == "right" and audio_high == "right" else "left"
            if response == correct_response:
                correct_answers += 1

            # Print trial details
            print(f"\nTrial {trial_index + 1} (Block {block}):")
            print(f"  Visual High: {visual_high}, Audio High: {audio_high}")
            print(
                f"  Correct Response: {correct_response}, User Response: {response}\n")

        self.block_scores.append(correct_answers)
        trigger_sender.send_trigger(29)  # block_end
        return correct_answers

    def run_experiment(self, block_focus_map):
        """
        Run the entire experiment.
        :param block_focus_map: Dictionary defining the focus type for each block
        """
        for block in range(1, self.config.NUM_BLOCKS + 1):
            focus_type = block_focus_map[block]
            correct_answers = self.run_block(block, focus_type)
            if block < self.config.NUM_BLOCKS:
                self.show_break(correct_answers, block)

    def show_break(self, correct_answers, block):
        """
        Display a break screen with performance feedback.
        :param correct_answers: Number of correct answers in the previous block
        :param block: The current block number
        """
        break_text = visual.TextStim(
            win, text=f"Block {block} complete. Take a 1-minute break.", color="white", pos=(0, 0)
        )
        break_text.draw()
        win.flip()
        core.wait(2)  # Show completion text for 2 seconds
        show_break_with_progress(
            win, self.config.BREAK_TIME, correct_answers, self.config.TRIALS_PER_BLOCK)


# Define focus for each block
block_focus_map = {
    1: "None",
    2: "Visual first, then Auditory",
    3: "Auditory first, then Visual",
    4: "None",
    5: "Auditory first, then Visual",
    6: "Visual first, then Auditory",
}


if __name__ == "__main__":
    # print(f"Detected Refresh Rate: {win.getActualFrameRate()}")

    # Initialize the trigger sender (assuming ParallelPort is already set up)
    trigger_sender = BittiumTriggerSender(
        ParallelPort(0x378), trigger_duration=0.005)

    # Experiment main loop
    trigger_sender.send_trigger(1)  # session_start

    # Show pre-experiment checklist
    show_pre_experiment_check(win)

    # Show introduction screen and wait for key press to continue
    show_introduction(win)

    # Resting State (adjustable duration)
    # Run resting state experiment
    # resting_state_eyes_open = RestingState(
    #     win, trigger_sender, duration=1, eyes_open=True)
    # resting_state_eyes_open.start()  # Eye open phase

    # resting_state_eyes_closed = RestingState(
    #     win, trigger_sender, duration=1, eyes_open=False)
    # resting_state_eyes_closed.start()  # Eye closed phase

    # transition_text = visual.TextStim(
    #     win, text="Resting state phase is complete.\n\n Press any key to begin the next phase.", color="white", pos=(0, 0))
    # transition_text.draw()
    # win.flip()
    # event.waitKeys()  # Wait for any key press

    # Visual Training
    training_text = visual.TextStim(
        win,
        text=("Resting state phase is complete, welcome to the training session.\n\n"
              "This training will help you become familiar with different visual and auditory frequencies,\n"
              "which will be important for the main experiment.\n\n"
              "We will start with the visual training session first.\n\n"
              "In the visual training session, you need to focus on the visual stimuli from different side.\n\n"
              "Press any key to begin the visual training session."),
        color="white", pos=(0, 0), wrapWidth=1200,
    )
    training_text.draw()
    win.flip()
    event.waitKeys()

    # Run visual training sessions
    trigger_sender.send_trigger(10)  # visual_training_start
    training_manager = TrainingManager(win, trigger_sender)
    config = ExperimentConfig()
    training_manager.run_visual_training(config.VISUAL_FREQ_HIGH, config.VISUAL_FREQ_LOW, 10,
                                         "Focus on the left-side high-frequency stimuli.")
    # training_manager.run_visual_training(config.VISUAL_FREQ_LOW, config.VISUAL_FREQ_HIGH, 1,
    #                                      "Focus on the left-side low-frequency stimuli.")
    # training_manager.run_visual_training(config.VISUAL_FREQ_LOW, config.VISUAL_FREQ_HIGH, 1,
    #                                      "Focus on the right-side high-frequency stimuli.")
    # training_manager.run_visual_training(config.VISUAL_FREQ_HIGH, config.VISUAL_FREQ_LOW, 1,
    #                                      "Focus on the right-side low-frequency stimuli.")
    trigger_sender.send_trigger(11)  # visual_training_end

    # Auditory Training
    training_text = visual.TextStim(
        win,
        text=("Visual traning session is complete.\n\n"
              "You will now begin the auditory training session.\n\n"
              "In the auditory training session, you need to focus on the auditory stimuli from different side.\n\n"
              "Press any key to begin the auditory training session."),
        color="white", pos=(0, 0), wrapWidth=1200,
    )
    training_text.draw()
    win.flip()
    event.waitKeys()

    # Run auditory training sessions
    trigger_sender.send_trigger(12)  # auditory_training_start
    # training_manager.run_auditory_training(
    #     tone_high, tone_low, tone_high_freq, 10,
    #     "Focus on the left-side high-frequency auditory stimuli."
    # )
    # training_manager.run_auditory_training(
    #     tone_low, tone_high, tone_low_freq, 1,
    #     "Focus on the left-side low-frequency auditory stimuli."
    # )
    # training_manager.run_auditory_training(
    #     tone_low, tone_high, tone_low_freq, 1,
    #     "Focus on the right-side high-frequency auditory stimuli."
    # )
    # training_manager.run_auditory_training(
    #     tone_high, tone_low, tone_high_freq, 1,
    #     "Focus on the right-side low-frequency auditory stimuli."
    # )
    trigger_sender.send_trigger(13)  # auditory_training_end

    # Transition to Main Experiment
    # ready_text = visual.TextStim(
    #     win,
    #     text="Training session is complete.\n\nYou will now begin the main experiment.\n\nPress any key to begin the experiment.",
    #     color="white"
    # )
    # ready_text.draw()
    # win.flip()
    # event.waitKeys()

    # Introduction Screens
    intro_text = visual.TextStim(
        win,
        text=("Training session is complete. Welcome to the experiment.\n\n"
              "Your task is to identify which visual and auditory stimuli "
              "have the higher frequency.\n\n"
              "If the higher frequency stimuli of visual and auditory both are on the right side, press the 'right' key.\n"
              "Otherwise, press the 'left' key.\n\n"
              "Press any key to start when you are ready."),
        color="white", pos=(0, 0), wrapWidth=1200,
    )
    intro_text.draw()
    win.flip()
    event.waitKeys()

    stimuli_combinations = [
        {"visual_high": "right", "audio_high": "right"},
        {"visual_high": "right", "audio_high": "left"},
        {"visual_high": "left", "audio_high": "right"},
        {"visual_high": "left", "audio_high": "left"},
    ]

    block_scores = []

    experiment = Experiment(
        win=win,
        config=config,
        trigger_sender=trigger_sender,
        stimuli_combinations=stimuli_combinations,
        stereo_combinations=stereo_combinations,
        block_scores=block_scores
    )
    experiment.run_experiment(block_focus_map=block_focus_map)

    trigger_sender.send_trigger(255)  # session_end

    # End of experiment
    # Calculate total score
    total_score = sum(block_scores)

    # Display scores for each block and total score
    score_text = "\n".join(
        [f"Block {i + 1}: {score}/{ExperimentConfig.TRIALS_PER_BLOCK}" for i, score in enumerate(block_scores)])
    final_score_text = f"Total Score: {total_score}/{ExperimentConfig.NUM_BLOCKS * ExperimentConfig.TRIALS_PER_BLOCK}\n\n" + score_text

    end_score_text = visual.TextStim(
        win,
        text=("Experiment complete.\n\n"
              "Here is your performance summary:\n\n"
              f"{final_score_text}\n\n"
              "Thank you for participating.\n\n"
              "You will now go through the resting state phase again before the experiment end.\n\n"
              "Please relax and focus.\n\n"
              "Press any key to begin."),
        color="white", pos=(0, 0)
    )
    end_score_text.draw()
    win.flip()
    event.waitKeys()  # Wait for participant to acknowledge

    # Run resting state experiment
    resting_state_eyes_open = RestingState(
        win, trigger_sender, duration=5, eyes_open=True)
    resting_state_eyes_open.start()  # Eye open phase

    resting_state_eyes_closed = RestingState(
        win, trigger_sender, duration=5, eyes_open=False)
    resting_state_eyes_closed.start()  # Eye closed phase

    transition_text = visual.TextStim(
        win, text="Resting state phase is complete.\n\n Goodbye.", color="white", pos=(0, 0))
    transition_text.draw()
    win.flip()
    core.wait(5)
    win.close()
