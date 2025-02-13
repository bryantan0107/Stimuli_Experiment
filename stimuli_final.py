from psychopy import visual, core, event
from psychopy.parallel import ParallelPort
import numpy as np
import random
import sounddevice as sd
import sys
import threading


# Experiment main configuration
class ExperimentConfig:
    """
    Configuration parameters for the experiment.
    """
    SAMPLE_RATE = 48000
    TONE_DURATION = 5  # For create_modulated_tone method
    CARRIER_FREQ_HIGH = 450
    CARRIER_FREQ_LOW = 400
    MOD_FREQ_HIGH = 40
    MOD_FREQ_LOW = 25
    VISUAL_FREQ_HIGH = 15  # Other options: 6, 7.5, 10, 12, 15, 20, 30, 60
    VISUAL_FREQ_LOW = 12
    RESTING_DURATION = 60
    TRAINING_DURATION = 15
    NUM_BLOCKS = 6
    TRIALS_PER_BLOCK = 40
    BREAK_TIME = 60
    REFRESH_RATE = 60  # Screen refresh rate (Hz)
    RUN_RESTING_STATE = True  # When False the resting state will be skipped
    RUN_TRAINING_SESSION = True  # When False the training session will be skipped


# Visual and Auditory stimulus combinations for main experiment
stimuli_combinations = [
    {"visual_high": "right", "audio_high": "right"},
    {"visual_high": "right", "audio_high": "left"},
    {"visual_high": "left", "audio_high": "right"},
    {"visual_high": "left", "audio_high": "left"},
]

# Focus conditions for blocks
block_focus_map = {
    1: "None",
    2: "Visual first, then Auditory",
    3: "Auditory first, then Visual",
    4: "None",
    5: "Auditory first, then Visual",
    6: "Visual first, then Auditory",
}

# PsychoPy window setup
win = visual.Window(
    size=(1440, 900),
    color=(0, 0, 0),  # Background color (black)
    units="pix",
    fullscr=True,  # Enable full screen
    allowGUI=False,  # Disable GUI elements
    screen=1  # Target screen
)

# Audio Manager for creating tone


class AudioManager:
    """
    Manages tone creation for the experiment.
    """
    @staticmethod
    def create_modulated_tone(mod_freq, carrier_freq, mod_depth):
        """
        Create an amplitude-modulated tone.
        :param mod_freq: Modulation frequency
        :param carrier_freq: Carrier frequency
        :param mod_depth: Modulation depth
        :return: Tuple containing the generated tone and its modulation frequency
        """
        time = np.linspace(0, ExperimentConfig.TONE_DURATION, int(
            ExperimentConfig.SAMPLE_RATE * ExperimentConfig.TONE_DURATION), endpoint=False)
        am_wave = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * time)) * \
            np.sin(2 * np.pi * carrier_freq * time)
        am_wave /= np.max(np.abs(am_wave))  # Normalize amplitude to [-1, 1]
        return am_wave.astype(np.float32), mod_freq


# Generate high and low auditory stimuli with corresponding modulation frequency
tone_low, tone_low_freq = AudioManager.create_modulated_tone(
    ExperimentConfig.MOD_FREQ_LOW, ExperimentConfig.CARRIER_FREQ_LOW, 1)
tone_high, tone_high_freq = AudioManager.create_modulated_tone(
    ExperimentConfig.MOD_FREQ_HIGH, ExperimentConfig.CARRIER_FREQ_HIGH, 1)

# Stereo combinations for auditory stimuli
stereo_combinations = {
    "right_high_left_low": (tone_high, tone_low),
    "right_low_left_high": (tone_low, tone_high),
}

# Utility function to check for escape key


def check_escape(win):
    """
    Checks if the 'escape' key is pressed to quit the experiment.
    :param win: PsychoPy window instance
    """
    keys = event.getKeys(keyList=["escape"])
    if "escape" in keys:
        win.close()
        core.quit()

# Trigger Sender Class


class BittiumTriggerSender:
    """
    Sends triggers to external hardware. (Bittium NeurOne EEG system)
    """

    def __init__(self, parallel_port: ParallelPort, trigger_duration: float) -> None:
        self.parallel_port = parallel_port
        self.trigger_duration = trigger_duration

    def send_trigger(self, trigger_code: int) -> None:
        """
        Sends a trigger to the hardware.
        :param trigger_code: Trigger code to be sent
        """
        if sys.platform == "darwin":  # For macOS testing, because it does not support parallel port
            print(f"Trigger sent: {trigger_code}")
        else:
            try:
                print(f"Trigger sent: {trigger_code}")
                clock = core.Clock()
                self.parallel_port.setData(trigger_code)
                while clock.getTime() < self.trigger_duration:
                    pass
                self.parallel_port.setData(0)  # Reset trigger to low
                while clock.getTime() < self.trigger_duration:
                    pass
            except Exception as e:
                print(f"Error sending trigger: {e}")


# Function to check experimental setup


def show_pre_experiment_check(win, trigger_sender, left_tone, right_tone, left_tone_freq):
    """
    Display a pre-experiment checklist to ensure setup is ready, and allow toggling auditory stimulus playback.
    :param win: PsychoPy window instance
    :param trigger_sender: Trigger sender instance for auditory stimulus
    :param left_tone: Audio waveform for the left ear
    :param right_tone: Audio waveform for the right ear
    :param left_tone_freq: Frequency of the left tone
    """
    # Initialize the auditory stimulus
    auditory_stimulus = AuditoryStimulus(
        trigger_sender, left_tone, right_tone, left_tone_freq)

    # Define the checklist text
    checklist_text = visual.TextStim(
        win,
        text=("Hello, and welcome!\n\n"
              "Before we begin, please let the experimenter confirm the following:\n\n"
              "1. The keyboard is correctly positioned and ready to use.\n"
              "2. The audio volume is set to a comfortable level.\n\n"
              "Press 'Enter' to play/pause the audio.\n"
              "Press 'Space' to proceed to the next step."),
        color="white",
        pos=(0, 0),
        wrapWidth=1200,
    )

    # Display the checklist text
    checklist_text.draw()
    win.flip()

    while True:
        keys = event.getKeys()
        if "space" in keys:  # Proceed to the next step
            break
        if "return" in keys:  # Toggle auditory stimulus playback
            auditory_stimulus.present(duration=3)

        # Redraw the checklist text
        checklist_text.draw()
        win.flip()

    return


# Introduction Screen after pre-experiment checklist
def show_introduction(win):
    """
    Displays the experiment introduction screen.
    """
    intro_text = visual.TextStim(
        win,
        text="Welcome to the experiment!\n\n"
             "We will now begin with a resting state phase.\n\n"
             "During this phase, you may either keep your eyes open or close them.\n\n"
             "Please take this time to relax and focus.\n\n"
             "When you are ready, press any key to begin.",
        color="white", pos=(0, 0), wrapWidth=1200,
    )
    intro_text.draw()
    win.flip()
    event.waitKeys()

# Countdown Timer


def countdown(win, rest_text):
    countdown_text = visual.TextStim(
        win,
        color="white",
        pos=(0, -100),
    )

    countdown_time = 4
    start_time = core.getTime()
    while True:
        elapsed_time = core.getTime() - start_time
        remaining_time = countdown_time - elapsed_time

        if remaining_time <= 1:
            break

        countdown_text.text = f"{int(remaining_time)}"

        rest_text.draw()
        countdown_text.draw()
        win.flip()

# Resting State Class


class RestingState:
    """
    Manages the resting state display and trigger sending.
    """

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
            self.win, text="", color="white", pos=(0, 0), wrapWidth=1200)

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
            self.rest_text.text = "You will see a cross displayed on the screen.\n\n Please focus on the cross and keep your eyes open."
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
            2 if self.eyes_open else 3)

        self.show_fixation()

        self.trigger_sender.send_trigger(
            4 if self.eyes_open else 5)

        # Play a sound only if the eyes were closed
        if not self.eyes_open:
            SAMPLE_RATE = 48000  # Sample rate for audio
            DURATION = 0.5  # Duration of the beep sound in seconds
            FREQUENCY = 600  # Frequency of the beep sound in Hz (A4 note)

            # Generate the beep sound
            time = np.linspace(0, DURATION, int(
                SAMPLE_RATE * DURATION), endpoint=False)
            beep = np.sin(2 * np.pi * FREQUENCY * time)

            sd.play(beep, samplerate=SAMPLE_RATE)
            sd.wait()


class Stimulus:
    def __init__(self, win):
        """
        Initialize the base Stimulus class.
        :param win: PsychoPy window instance
        :param duration: Duration of the stimulus in seconds
        """
        self.win = win

    def present(self):
        """Display the stimulus (to be implemented in subclasses)."""
        raise NotImplementedError("Subclasses must implement this method.")


class VisualStimulus(Stimulus):
    def __init__(self, win, trigger_sender, left_freq, right_freq):
        """
        Initialize the visual stimulus.
        :param win: PsychoPy window instance
        :param left_freq: Frequency for the left visual stimulus
        :param right_freq: Frequency for the right visual stimulus
        :param size: Size of the visual stimulus
        """
        super().__init__(win)
        self.left_freq = left_freq
        self.right_freq = right_freq
        self.trigger_sender = trigger_sender

        # Create left and right grating stimuli
        self.left_grating = visual.GratingStim(
            win, tex="sin", mask="gauss", sf=0.035, units="pix", size=(350, 350), pos=(-220, 0), ori=0, contrast=0.8)
        self.right_grating = visual.GratingStim(
            win, tex="sin", mask="gauss", sf=0.035, units="pix", size=(350, 350), pos=(220, 0), ori=0, contrast=0.8)

        # Fixation cross to maintain participant's focus
        self.fixation = visual.TextStim(
            self.win, text="+", color="white", height=70  # Fixation cross
        )

    def present(self, duration=None, response_handler=None, block_trial_text=None):
        """
        Present the visual stimulus with alternating flickers.
        :param duration: Duration to display the stimulus (None for indefinite).
        :param response_handler: External function to handle response.
        :param block_trial_text: Text object for displaying trial/block info.
        :return: User response if applicable.
        """
        # Calculate flicker timing based on refresh rate
        frames_per_cycle_left = int(
            ExperimentConfig.REFRESH_RATE / self.left_freq)
        frames_per_cycle_right = int(
            ExperimentConfig.REFRESH_RATE / self.right_freq)

        # Define flicker cycle for left and right stimuli
        on_frames_left = frames_per_cycle_left // 2
        off_frames_left = frames_per_cycle_left - on_frames_left
        on_frames_right = frames_per_cycle_right // 2
        off_frames_right = frames_per_cycle_right - on_frames_right

        # Initialize frame counters and states
        frame_n_left = 0
        frame_n_right = 0
        left_on = True
        right_on = True
        last_left_frame = frame_n_left
        last_right_frame = frame_n_right

        clock = core.Clock()
        start_time = clock.getTime()

        while duration is None or clock.getTime() - start_time < duration:
            # Check for user response
            keys = event.getKeys(keyList=["left", "right", "escape"])
            if keys:
                user_response = keys[0]
                if "escape" in keys:
                    self.win.close()
                    core.quit()
                elif user_response == "left":
                    self.trigger_sender.send_trigger(
                        25)  # Left button trigger
                elif user_response == "right":
                    self.trigger_sender.send_trigger(
                        26)  # Right button trigger
                break

            if response_handler and response_handler():  # Exit if external response handler is triggered
                break

            # Update left grating state
            if left_on and frame_n_left >= last_left_frame + on_frames_left:
                left_on = False
                last_left_frame = frame_n_left
            if not left_on and frame_n_left >= last_left_frame + off_frames_left:
                left_on = True
                last_left_frame = frame_n_left

            # Update right grating state
            if right_on and frame_n_right >= last_right_frame + on_frames_right:
                right_on = False
                last_right_frame = frame_n_right
            if not right_on and frame_n_right >= last_right_frame + off_frames_right:
                right_on = True
                last_right_frame = frame_n_right

            frame_n_left += 1
            frame_n_right += 1

            # Draw fixation cross
            self.fixation.draw()

            # Draw gratings based on current states
            if left_on:
                self.left_grating.draw()
            if right_on:
                self.right_grating.draw()

            win.flip()  # Update the window

            # Optionally draw block_trial_text
            if block_trial_text is not None:
                block_trial_text.draw()

        return None if duration is not None else user_response


class AuditoryStimulus(Stimulus):
    def __init__(self, trigger_sender, left_tone, right_tone, left_tone_freq):
        """
        Auditory stimulus consisting of stereo tones.
        :param trigger_sender: Object to send event triggers.
        :param left_tone: Audio waveform for the left ear.
        :param right_tone: Audio waveform for the right ear.
        :param left_tone_freq: Frequency of the left tone.
        """
        super().__init__(win=None)
        self.left_tone_freq = left_tone_freq
        self.trigger_sender = trigger_sender

        # Merge left and right tone signals into stereo format
        self.stereo_wave = np.column_stack((left_tone, right_tone)).astype(
            np.float32)  # Convert data type to float32

    def present(self, duration=None, response_handler=None):
        """
        Play the auditory stimulus.
        :param duration: Duration of stimulus playback (None for infinite).
        :param response_handler: External function to handle response.
        """
        stream = sd.OutputStream(
            samplerate=ExperimentConfig.SAMPLE_RATE, channels=2)

        try:
            stream.start()
            clock = core.Clock()
            audio_index = 0
            buffer_size = 1024  # Chunk size for playback

            while True:
                # Stop based on duration or external response handler
                if duration is not None and clock.getTime() >= duration:
                    break
                elif response_handler and response_handler():
                    break

                # Play audio in chunks
                if audio_index < len(self.stereo_wave):
                    stream.write(
                        self.stereo_wave[audio_index:audio_index + buffer_size])
                    audio_index += buffer_size
                else:
                    audio_index = 0  # Loop playback
        finally:
            stream.stop()
            stream.close()  # Ensure cleanup of audio stream


# Visual Training with flickering
class TrainingManager:
    def __init__(self, win, trigger_sender):
        self.win = win
        self.trigger_sender = trigger_sender

    def run_visual_training(self, left_freq, right_freq, duration, instruction_text, focus_side):
        # Show instruction
        training_text = visual.TextStim(
            self.win, text=instruction_text, color="white", wrapWidth=1200)
        training_text.draw()
        self.win.flip()
        core.wait(2)

        countdown(win, training_text)

        # Determine the trigger based on fast stimulus side and focus side
        fast_side = "left" if left_freq == ExperimentConfig.VISUAL_FREQ_HIGH else "right"

        trigger_map = {
            ("left", "left"): 10,  # Fast Left, Focus Left
            ("right", "left"): 11,  # Fast Right, Focus Left
            ("right", "right"): 12,  # Fast Right, Focus Right
            ("left", "right"): 13,  # Fast Left, Focus Right
        }

        trigger_code = trigger_map[(fast_side, focus_side)]
        self.trigger_sender.send_trigger(trigger_code)

        visual_stimulus = VisualStimulus(
            win, self.trigger_sender, left_freq, right_freq)
        visual_stimulus.present(duration)

    def run_auditory_training(self, left_tone, right_tone, left_tone_freq, duration, instruction_text, focus_side):
        """
        Runs auditory training with specified left and right tones for a given duration.
        :param left_tone: The audio waveform for the left ear
        :param right_tone: The audio waveform for the right ear
        :param duration: Duration of the training in seconds
        :param instruction_text: Instruction to display before training starts
        """
        # Show instruction
        training_text = visual.TextStim(
            self.win, text=instruction_text, color="white", wrapWidth=1200)
        training_text.draw()
        self.win.flip()
        core.wait(3)

        countdown(win, training_text)

        # Determine the trigger based on fast stimulus side and focus side
        fast_side = "left" if left_tone_freq == ExperimentConfig.MOD_FREQ_HIGH else "right"

        trigger_map = {
            ("left", "left"): 14,  # Fast Left, Focus Left
            ("right", "left"): 15,  # Fast Right, Focus Left
            ("right", "right"): 16,  # Fast Right, Focus Right
            ("left", "right"): 17,  # Fast Left, Focus Right
        }

        trigger_code = trigger_map[(fast_side, focus_side)]
        self.trigger_sender.send_trigger(trigger_code)

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
        :param block_scores: List to store scores for each block
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
        :param trials: List of trials to use for practice
        """
        practice_instruction_text = {
            "None": "This is a practice session to help you get familiar with the task.\n\n"
                    "When you are ready, press any key to start the practice session.",
            "Visual first, then Auditory": "This is a practice session to help you focus on the visual stimuli first, followed by the auditory stimuli.\n\n"
                                           "When you are ready, press any key to start the practice session.",
            "Auditory first, then Visual": "This is a practice session to help you focus on the auditory stimuli first, followed by the visual stimuli.\n\n"
                                           "When you are ready, press any key to start the practice session."
        }

        # Display instructions for the practice session
        instruction = visual.TextStim(
            self.win,
            text="Welcome to the practice session.\n\n" +
            practice_instruction_text[focus_type],
            color="white",
            pos=(0, 0),
            wrapWidth=1200
        )
        instruction.draw()
        self.win.flip()
        event.waitKeys()

        while True:
            # Randomly select a practice trial
            practice_trial = random.choice(trials)

            # Create and run a single practice trial
            response, visual_high, audio_high = self.run_trial(
                -1, practice_trial, 0)

            correct_response = "right" if visual_high == "right" and audio_high == "right" else "left"

            feedback = visual.TextStim(
                self.win,
                text=f"Visual High: {visual_high}, Audio High: {audio_high}\n"
                f"Correct Response: {correct_response}, Your Response: {response}\n\n"
                "Did you feel ready to complete the task?\n\n"
                "Press 'right' to proceed to the trial.\n"
                "Press 'left' to repeat the practice session.",
                font="Noto Color Emoji", color="white", pos=(0, 0)
            )
            feedback.draw()
            self.win.flip()

            keys = event.waitKeys(keyList=["left", "right"])
            if "right" in keys:
                break  # Exit the practice session

    def show_block_focus(self, block, focus_type):
        """
        Display instructions for a specific block based on the focus type.
        :param block: Current block number
        :param focus_type: The type of focus for this block
        """
        focus_messages = {
            "None": "There are no specific instructions for this block. Please proceed as usual.\n\n",
            "Visual first, then Auditory": "In this block, please focus on the visual stimuli first, followed by the auditory stimuli.\n\n",
            "Auditory first, then Visual": "In this block, please focus on the auditory stimuli first, followed by the visual stimuli.\n\n",
        }
        instruction_text = f"Block {block}/{ExperimentConfig.NUM_BLOCKS} Instructions:\n\n" + \
            focus_messages[focus_type]
        instruction_text += "When you are ready, press any key to start this block."

        # Display the instruction text
        block_instruction = visual.TextStim(
            win, text=instruction_text, color="white", pos=(0, 0), wrapWidth=1200)
        block_instruction.draw()
        win.flip()
        event.waitKeys()

    def show_rule_reminder(self, win, focus_type):
        """
        Display the block-specific focus instructions along with the general experiment rules.
        :param win: PsychoPy window instance
        :param block: Current block number
        :param focus_type: The type of focus for this block
        """
        # Define block-specific focus messages
        focus_messages = {
            "None": "There are no specific instructions for this block. Please proceed as usual.\n\n",
            "Visual first, then Auditory": "In this block, please focus on the visual stimuli first, followed by the auditory stimuli.\n\n",
            "Auditory first, then Visual": "In this block, please focus on the auditory stimuli first, followed by the visual stimuli.\n\n",
        }

        # Combine block-specific focus with general rules
        reminder_text = (
            "Reminder:\n\n "
            "This is a brief reminder to help you stay focused on the task:\n\n"
            "Your task is to determine which stimuli have the higher frequency.\n\n"
            "If both the higher frequency visual and auditory stimuli are on the right side, press the 'right' key.\n"
            "Otherwise, press the 'left' key.\n\n"
            "There will be a cross in the center of the screen. Please try your best to\n"
            "keep your eyes focused on the cross while observing the flickering patterns.\n\n"
            f"Here are the instructions for this block:\n\n"
            + focus_messages.get(focus_type, "")
            + "When you are ready, press any key to continue."
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
        Present both visual and auditory stimuli simultaneously and capture user response.
        :param visual_stimulus: Visual stimulus object
        :param auditory_stimulus: Auditory stimulus object
        :param block_trial_text: Instructional text for the trial
        :return: User response
        """

        # Flags to control the flow of the trial
        stop_flag = threading.Event()  # Signals when the trial should stop
        # Signals when the audio stimulus has finished playing
        audio_finished_flag = threading.Event()

        def response_handler():
            return stop_flag.is_set()

        def play_audio():
            """
            Function to play the auditory stimulus in a separate thread.
            It stops when response_handler returns True.
            """
            try:
                auditory_stimulus.present(
                    duration=None, response_handler=response_handler)
            except Exception as e:
                print(f"Error during audio playback: {e}")
            finally:
                audio_finished_flag.set()

        # Start a separate thread for audio playback to ensure simultaneous execution
        audio_thread = threading.Thread(target=play_audio, daemon=True)
        audio_thread.start()

        user_response = None

        try:
            user_response = visual_stimulus.present(
                duration=None,
                response_handler=lambda: stop_flag.is_set() or audio_finished_flag.is_set(),
                block_trial_text=block_trial_text
            )
            win.flip()

        finally:
            stop_flag.set()
            audio_thread.join()

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

        # Mapping visual and auditory high-frequency locations to trigger codes
        trigger_map = {
            ("left", "left"): 21,  # Visual Fast Left, Auditory Fast Left (S00)
            ("left", "right"): 22,  # Visual Fast Left, Auditory Fast Right (S01)
            ("right", "left"): 23,  # Visual Fast Right, Auditory Fast Left (S10)
            ("right", "right"): 24,  # Visual Fast Right, Auditory Fast Right (S11)
        }

        trigger_code = trigger_map[(visual_high, audio_high)]
        self.trigger_sender.send_trigger(trigger_code)

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

        # Present both stimuli and capture user response
        response = self.present_visual_and_auditory(
            visual_stimulus, auditory_stimulus, block_trial_text)
        win.flip()
        core.wait(0.5)

        # Show rule reminder every 10 trials
        if focus_type and (trial_index + 1) % 10 == 0 and (trial_index + 1) != self.config.TRIALS_PER_BLOCK:
            self.show_rule_reminder(self.win, focus_type)

        return response, visual_high, audio_high

    def run_block(self, block, focus_type):
        """
        Execute a single block of trials in the experiment.
        :param block: The current block number
        :param focus_type: The type of focus for this block
        :return: The number of correct responses in this block
        """
        trigger_sender.send_trigger(19)  # block_start

        self.show_block_focus(block, focus_type)

        correct_answers = 0

        # Create a randomized list of trials for this block
        trials = self.stimuli_combinations * \
            (self.config.TRIALS_PER_BLOCK // len(self.stimuli_combinations))
        random.shuffle(trials)

        # Run practice session before the block starts
        self.run_practice_session(focus_type, trials)

        for trial_index, trial in enumerate(trials):
            response, visual_high, audio_high = self.run_trial(
                trial_index, trial, block, focus_type)

            correct_response = "right" if visual_high == "right" and audio_high == "right" else "left"
            if response == correct_response:
                correct_answers += 1

            # Print trial summary for debugging/logging (if needed)
            print(f"\nTrial {trial_index + 1} (Block {block}):")
            print(f"Visual High: {visual_high}, Audio High: {audio_high}")
            print(
                f"  Correct Response: {correct_response}, User Response: {response}\n")

        # Store the correct response count for this block
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

    def show_break_with_progress(self, win, break_time, correct_answers, total_trials):
        progress_text = visual.TextStim(
            win,
            text="Please take a break.\n\nTime remaining: 60 seconds",
            color="white",
            pos=(0, 0),
        )
        score_text = visual.TextStim(
            win,
            text=f"Your results from the last block:\n {correct_answers}/{total_trials} correct answers.",
            color="white",
            pos=(0, 70),
            wrapWidth=1200,
        )

        start_time = core.getTime()
        while core.getTime() - start_time < break_time:
            elapsed_time = core.getTime() - start_time
            remaining_time = break_time - elapsed_time

            progress_text.text = f"Please take a break.\n\nTime remaining: {int(remaining_time)} seconds"

            score_text.draw()
            progress_text.draw()
            win.flip()

    def show_break(self, correct_answers, block):
        """
        Display a break screen with performance feedback.
        :param correct_answers: Number of correct answers in the previous block
        :param block: The current block number
        """
        break_text = visual.TextStim(
            win, text=f"Block {block} complete. Take a short 1-minute break.", color="white", pos=(0, 0)
        )
        break_text.draw()
        win.flip()
        core.wait(3)
        self.show_break_with_progress(
            win, self.config.BREAK_TIME, correct_answers, self.config.TRIALS_PER_BLOCK)


if __name__ == "__main__":
    # Initialize the trigger sender (assuming ParallelPort is already set up)
    trigger_sender = BittiumTriggerSender(
        ParallelPort(0x378), trigger_duration=0.005)

    # Experiment main loop
    trigger_sender.send_trigger(1)  # session_start

    # Show pre-experiment checklist
    show_pre_experiment_check(
        win, trigger_sender, tone_high, tone_low, tone_high_freq)

    # Show introduction screen and wait for key press to continue
    show_introduction(win)

    if ExperimentConfig.RUN_RESTING_STATE == True:
        # Run resting state experiment
        resting_state_eyes_open = RestingState(
            win, trigger_sender, ExperimentConfig.RESTING_DURATION, eyes_open=True)
        resting_state_eyes_open.start()  # Eye open phase
        resting_state_eyes_closed = RestingState(
            win, trigger_sender, ExperimentConfig.RESTING_DURATION, eyes_open=False)
        resting_state_eyes_closed.start()  # Eye closed phase

    # Visual Training
    training_text = visual.TextStim(
        win,
        text=("Resting state phase complete. Welcome to the training session!\n\n"
              "This training session is designed to familiarize you with the stimuli that will be\n"
              "important for the main experiment.\n\n"
              "We will begin with the visual training session.\n\n"
              "In this session, your task is to focus on visual stimuli presented on different sides.\n\n"
              "There will be a cross in the center of the screen. Please try your best to\n"
              "keep your eyes focused on the cross while observing the flickering patterns.\n\n"
              "When you are ready, press any key to start the visual training session."),
        color="white", pos=(0, 0), wrapWidth=1200,
    )
    training_text.draw()
    win.flip()
    event.waitKeys()

    # Run visual training sessions
    if ExperimentConfig.RUN_TRAINING_SESSION == True:
        training_manager = TrainingManager(win, trigger_sender)
        training_manager.run_visual_training(ExperimentConfig.VISUAL_FREQ_HIGH, ExperimentConfig.VISUAL_FREQ_LOW, ExperimentConfig.TRAINING_DURATION,
                                             "Focus on the left-side high-frequency stimuli.", "left")
        training_manager.run_visual_training(ExperimentConfig.VISUAL_FREQ_LOW, ExperimentConfig.VISUAL_FREQ_HIGH, ExperimentConfig.TRAINING_DURATION,
                                             "Focus on the left-side low-frequency stimuli.", "left")
        training_manager.run_visual_training(ExperimentConfig.VISUAL_FREQ_LOW, ExperimentConfig.VISUAL_FREQ_HIGH, ExperimentConfig.TRAINING_DURATION,
                                             "Focus on the right-side high-frequency stimuli.", "right")
        training_manager.run_visual_training(ExperimentConfig.VISUAL_FREQ_HIGH, ExperimentConfig.VISUAL_FREQ_LOW, ExperimentConfig.TRAINING_DURATION,
                                             "Focus on the right-side low-frequency stimuli.", "right")

    # Auditory Training
    training_text = visual.TextStim(
        win,
        text=("Visual training session complete.\n\n"
              "We will now move on to the auditory training session.\n\n"
              "In this session, your task is to focus on auditory stimuli presented to different sides.\n\n"
              "Press any key to begin the auditory training session."),
        color="white", pos=(0, 0), wrapWidth=1200,
    )
    training_text.draw()
    win.flip()
    event.waitKeys()

    # Run auditory training sessions
    if ExperimentConfig.RUN_TRAINING_SESSION == True:
        training_manager.run_auditory_training(
            tone_high, tone_low, tone_high_freq, ExperimentConfig.TRAINING_DURATION,
            "Focus on the left-side high-frequency auditory stimuli.", "left"
        )
        training_manager.run_auditory_training(
            tone_low, tone_high, tone_low_freq, ExperimentConfig.TRAINING_DURATION,
            "Focus on the left-side low-frequency auditory stimuli.", "left"
        )
        training_manager.run_auditory_training(
            tone_low, tone_high, tone_low_freq, ExperimentConfig.TRAINING_DURATION,
            "Focus on the right-side high-frequency auditory stimuli.", "right"
        )
        training_manager.run_auditory_training(
            tone_high, tone_low, tone_high_freq, ExperimentConfig.TRAINING_DURATION,
            "Focus on the right-side low-frequency auditory stimuli.", "right"
        )

    # Introduction Screens
    intro_text = visual.TextStim(
        win,
        text=("Training session complete. Welcome to the main experiment!\n\n"
              "Your task is as follows:\n\n"
              "If both the higher frequency visual and auditory stimuli are on the right side, press the 'right' key.\n"
              "If not, press the 'left' key.\n\n"
              "There will be a cross in the center of the screen. Please try your best to\n"
              "keep your eyes focused on the cross while observing the flickering patterns.\n\n"
              "When you are ready, press any key to start the experiment."),
        color="white", pos=(0, 0), wrapWidth=1200,
    )
    intro_text.draw()
    win.flip()
    event.waitKeys()

    block_scores = []

    experiment = Experiment(
        win=win,
        config=ExperimentConfig(),
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
    final_score_text = f"Overall Score: {total_score}/{ExperimentConfig.NUM_BLOCKS * ExperimentConfig.TRIALS_PER_BLOCK}\n\n" + score_text

    end_score_text = visual.TextStim(
        win,
        text=("Experiment complete. Thank you for your participation!\n\n"
              "Here is your overall performance summary:\n\n"
              f"{final_score_text}\n\n"
              "Before concluding, we will conduct one more resting state phase.\n\n"
              "Please relax and focus.\n\n"
              "Press any key to begin."),
        color="white", pos=(0, 0), wrapWidth=1200
    )
    end_score_text.draw()
    win.flip()
    event.waitKeys()  # Wait for participant to acknowledge

    # Run resting state experiment
    resting_state_eyes_open = RestingState(
        win, trigger_sender, ExperimentConfig.RESTING_DURATION, eyes_open=True)
    resting_state_eyes_open.start()

    resting_state_eyes_closed = RestingState(
        win, trigger_sender, ExperimentConfig.RESTING_DURATION, eyes_open=False)
    resting_state_eyes_closed.start()

    transition_text = visual.TextStim(
        win, text="Resting state phase is complete.\n\nThank you, and goodbye!", color="white", pos=(0, 0), wrapWidth=1200)
    transition_text.draw()
    win.flip()
    core.wait(10)
    win.close()
