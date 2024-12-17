from psychopy import visual, sound, core, event
from psychopy.parallel import ParallelPort
import numpy as np
import random
import sounddevice as sd
import sys


# Parameters
SAMPLE_RATE = 48000  # Sampling rate for audio
DURATION = 2  # Duration of each sound segment in seconds
CARRIER_FREQ = 440  # Carrier frequency in Hz
MOD_FREQ_HIGH = 40  # High modulation frequency in Hz
MOD_FREQ_LOW = 25  # Low modulation frequency in Hz
NUM_BLOCKS = 6
TRIALS_PER_BLOCK = 4
BREAK_TIME = 5  # Break duration in seconds
VISUAL_FREQ_HIGH = 18  # High visual flicker frequency in Hz
VISUAL_FREQ_LOW = 13  # Low visual flicker frequency in Hz
SF = 0.05  # Spatial frequency for grating
PRACTICE_TRIALS = 5  # Number of practice trials
REFRESH_RATE = 60  # Screen refresh rate in Hz

# Function to create modulated tones


def create_modulated_tone(mod_freq):
    time = np.linspace(0, DURATION, int(
        SAMPLE_RATE * DURATION), endpoint=False)
    am_wave = (1 + np.sin(2 * np.pi * mod_freq * time)) * \
        np.sin(2 * np.pi * CARRIER_FREQ * time)
    am_wave /= np.max(np.abs(am_wave))  # Normalize wave amplitude
    return am_wave.astype(np.float32)


# Precompute tones
tone_high = create_modulated_tone(MOD_FREQ_HIGH)
tone_low = create_modulated_tone(MOD_FREQ_LOW)

# Define stereo audio combinations
stereo_combinations = {
    "right_high_left_low": (tone_high, tone_low),
    "right_low_left_high": (tone_low, tone_high),
}


# Initialize PsychoPy window
win = visual.Window(size=(800, 600), color=(
    0, 0, 0), units="pix", fullscr=False)


# Create visual stimuli
# left_grating = visual.Rect(
#     win, width=100, height=100, pos=(-200, 0), fillColor="white"
# )
# right_grating = visual.Rect(
#     win, width=100, height=100, pos=(200, 0), fillColor="white"
# )

left_grating = visual.GratingStim(
    win, tex="sin", mask="circle", sf=SF, size=200, pos=(-200, 0))
right_grating = visual.GratingStim(
    win, tex="sin", mask="circle", sf=SF, size=200, pos=(200, 0))

# Stimuli combinations
stimuli_combinations = [
    {"visual_high": "right", "audio_high": "right"},
    {"visual_high": "right", "audio_high": "left"},
    {"visual_high": "left", "audio_high": "right"},
    {"visual_high": "left", "audio_high": "left"},
]

# Trigger sender class


class BittiumTriggerSender:
    def __init__(self, parallel_port: ParallelPort, trigger_duration: float) -> None:
        self.parallel_port = parallel_port
        self.trigger_duration = trigger_duration

    def send_trigger(self, trigger_code: int) -> None:
        if sys.platform == "darwin":  # macOS
            print(f"Trigger sent: {trigger_code}")
        else:
            # Send the trigger (setData sets the port pins to the value of trigger)
            self.parallel_port.setData(trigger_code)
            # Keep the trigger high for the specified time
            core.wait(self.trigger_duration)
            self.parallel_port.setData(0)  # Reset trigger to low
            # Small pause after resetting the trigger
            core.wait(self.trigger_duration)

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
        self.fixation.draw()
        self.win.flip()

    def update_rest_text(self):
        """Update resting state text based on eye state"""
        if self.state == "Eyes Open":
            self.rest_text.text = "Please look at the cross and keep your eyes open."
        else:
            self.rest_text.text = "Now close your eye."

    def start(self):
        """Start the resting state and send triggers"""
        self.update_rest_text()
        self.trigger_sender.send_trigger(
            2 if self.eyes_open else 3)  # RS_start_EO or RS_start_EC
        self.rest_text.draw()
        self.win.flip()
        core.wait(3)
        self.show_fixation()  # Show fixation cross
        core.wait(self.duration)
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

    def set_duration(self, duration):
        """Optionally change the duration of the resting state"""
        self.duration = duration

    def set_eyes_open(self, eyes_open):
        """Optionally change whether the eyes are open or closed"""
        self.eyes_open = eyes_open
        self.state = "Eyes Open" if eyes_open else "Eyes Closed"

# Function to display introduction text and wait for a key press


def show_introduction(win):
    intro_text = visual.TextStim(
        win,
        text="Welcome.\n\n"
             "You will now go through a resting state phase before the experiment start.\n\n"
             "Please relax and focus.\n\n"
             "Press any key to begin.",
        color="white", pos=(0, 0)
    )
    intro_text.draw()
    win.flip()
    event.waitKeys()  # Wait for any key press


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
        text=f"Performance: {correct_answers}/{total_trials} correct answers",
        color="white",
        pos=(0, 70),
    )

    start_time = core.getTime()
    while core.getTime() - start_time < break_time:
        elapsed_time = core.getTime() - start_time
        remaining_time = break_time - elapsed_time
        percentage = elapsed_time / break_time

        # Update progress bar width
        progress_text.text = f"Please take a break.\n\nTime remaining: {int(remaining_time)} seconds"

        # Draw performance text, progress bar, and countdown timer
        score_text.draw()
        progress_text.draw()
        win.flip()


# Visual Training with flickering


def run_visual_training(win, left_freq, right_freq, duration, instruction_text):
    """
    Runs visual training with specified left and right frequencies for a given duration.
    :param win: PsychoPy window
    :param left_freq: Frequency of left visual stimuli
    :param right_freq: Frequency of right visual stimuli
    :param duration: Duration of the training in seconds
    :param instruction_text: Instruction to display before training starts
    """
    # Show instruction
    training_text = visual.TextStim(win, text=instruction_text, color="white")
    training_text.draw()
    win.flip()
    core.wait(5)

    # Frames per cycle calculation for flickering
    frames_per_cycle_left = int(REFRESH_RATE / left_freq)
    frames_per_cycle_right = int(REFRESH_RATE / right_freq)
    on_frames_left = frames_per_cycle_left // 2
    off_frames_left = frames_per_cycle_left - on_frames_left
    on_frames_right = frames_per_cycle_right // 2
    off_frames_right = frames_per_cycle_right - on_frames_right

    # Initialize frame counters and states
    frame_n = 0
    left_on = True
    right_on = True
    last_left_frame = 0
    last_right_frame = 0

    # Run the visual training loop
    clock = core.Clock()
    start_time = clock.getTime()
    while clock.getTime() - start_time < duration:
        # Update left grating state
        if left_on and frame_n >= last_left_frame + on_frames_left:
            left_on = False
            last_left_frame = frame_n
        if not left_on and frame_n >= last_left_frame + off_frames_left:
            left_on = True
            last_left_frame = frame_n

        # Update right grating state
        if right_on and frame_n >= last_right_frame + on_frames_right:
            right_on = False
            last_right_frame = frame_n
        if not right_on and frame_n >= last_right_frame + off_frames_right:
            right_on = True
            last_right_frame = frame_n

        # Draw gratings based on current states
        if left_on:
            left_grating.draw()
        if right_on:
            right_grating.draw()

        win.flip()  # Update the window
        frame_n += 1


def run_auditory_training(left_tone, right_tone, duration, instruction_text):
    """
    Runs auditory training with specified left and right tones for a given duration.
    :param left_tone: The audio waveform for the left ear
    :param right_tone: The audio waveform for the right ear
    :param duration: Duration of the training in seconds
    :param instruction_text: Instruction to display before training starts
    """
    # Show instruction
    training_text = visual.TextStim(win, text=instruction_text, color="white")
    training_text.draw()
    win.flip()
    core.wait(5)

    # Create stereo audio wave
    stereo_wave = np.column_stack((left_tone, right_tone)).astype(np.float32)

    # Start non-blocking audio stream
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2)
    stream.start()

    clock = core.Clock()
    start_time = clock.getTime()
    audio_index = 0  # Starting index for audio playback
    buffer_size = 1024  # Buffer size for audio playback

    while clock.getTime() - start_time < duration:
        # Stream audio dynamically
        if audio_index < len(stereo_wave):
            stream.write(stereo_wave[audio_index: audio_index + buffer_size])
            audio_index += buffer_size
        else:
            audio_index = 0  # Reset audio index to loop sound

    stream.stop()
    stream.close()


def show_block_focus(win, block_num, focus_type):
    """
    Show instructions for the current block based on the focus type.
    """
    focus_messages = {
        "None": "\n\n",
        "Visual first, then Auditory": "In this block, please focus on the visual stimuli first.\n\n",
        "Auditory first, then Visual": "In this block, please focus on the auditory stimuli first.\n\n",
    }
    instruction_text = f"Block {block_num}/{NUM_BLOCKS} Instructions:\n\n" + \
        focus_messages[focus_type]
    instruction_text += "Press any key to start this block."

    # Display the instruction text
    block_instruction = visual.TextStim(
        win, text=instruction_text, color="white", pos=(0, 0))
    block_instruction.draw()
    win.flip()
    event.waitKeys()  # Wait for participant to acknowledge


# Define focus for each block
block_focus_map = {
    1: "None",
    2: "Visual first, then Auditory",
    3: "Auditory first, then Visual",
    4: "None",
    5: "Auditory first, then Visual",
    6: "Visual first, then Auditory",
}


# Initialize the trigger sender (assuming ParallelPort is already set up)
trigger_sender = BittiumTriggerSender(
    ParallelPort(0x378), trigger_duration=0.005)

# Experiment main loop
trigger_sender.send_trigger(1)  # session_start

# Show introduction screen and wait for key press to continue
show_introduction(win)

# Resting State (adjustable duration)
# Run resting state experiment
resting_state_eyes_open = RestingState(
    win, trigger_sender, duration=3, eyes_open=True)
resting_state_eyes_open.start()  # Eye open phase

resting_state_eyes_closed = RestingState(
    win, trigger_sender, duration=3, eyes_open=False)
resting_state_eyes_closed.start()  # Eye closed phase

transition_text = visual.TextStim(
    win, text="Resting state phase is complete.\n\n Press any key to begin the next phase.", color="white", pos=(0, 0))
transition_text.draw()
win.flip()
event.waitKeys()  # Wait for any key press

# Visual Training
training_text = visual.TextStim(
    win,
    text=("Welcome to the training session.\n\n"
          "This training will help you become familiar with different visual and auditory frequencies, "
          "which will be important for the main experiment.\n\n"
          "We will start with the visual training session first.\n\n"
          "Press any key to begin the visual training session."),
    color="white", pos=(0, 0)
)
training_text.draw()
win.flip()
event.waitKeys()

# Run visual training sessions
run_visual_training(win, VISUAL_FREQ_HIGH, VISUAL_FREQ_LOW, 5,
                    "Focus on the left-side high-frequency stimuli.")
run_visual_training(win, VISUAL_FREQ_LOW, VISUAL_FREQ_HIGH, 5,
                    "Focus on the left-side low-frequency stimuli.")
run_visual_training(win, VISUAL_FREQ_LOW, VISUAL_FREQ_HIGH, 5,
                    "Focus on the right-side high-frequency stimuli.")
run_visual_training(win, VISUAL_FREQ_HIGH, VISUAL_FREQ_LOW, 5,
                    "Focus on the right-side low-frequency stimuli.")


# Auditory Training
training_text = visual.TextStim(
    win,
    text=("Visual traning session is complete.\n\n"
          "You will now begin the auditory training session.\n\n"
          "Press any key to begin the auditory training session."),
    color="white", pos=(0, 0)
)
training_text.draw()
win.flip()
event.waitKeys()

# Run auditory training sessions
run_auditory_training(
    tone_high, tone_low, 5,
    "Focus on the left-side high-frequency auditory stimuli."
)
run_auditory_training(
    tone_low, tone_high, 5,
    "Focus on the left-side low-frequency auditory stimuli."
)
run_auditory_training(
    tone_low, tone_high, 5,
    "Focus on the right-side high-frequency auditory stimuli."
)
run_auditory_training(
    tone_high, tone_low, 5,
    "Focus on the right-side low-frequency auditory stimuli."
)

# Transition to Main Experiment
ready_text = visual.TextStim(
    win,
    text="Training session is complete.\n\nYou will now begin the main experiment.\n\nPress any key to begin the experiment.",
    color="white"
)
ready_text.draw()
win.flip()
event.waitKeys()

# Introduction Screens
intro_text = visual.TextStim(
    win,
    text=("Welcome to the experiment.\n\n"
          "Your task is to identify which visual and auditory stimuli "
          "have the higher frequency.\n\n"
          "If the higher frequency stimuli of visual and auditory both are on the right side, press the 'right' key.\n"
          "Otherwise, press the 'left' key.\n\n"
          "Press any key to start when you are ready."),
    color="white", pos=(0, 0)
)
intro_text.draw()
win.flip()
event.waitKeys()

# Initialize block scores
block_scores = []

# Experiment main loop
for block in range(1, NUM_BLOCKS + 1):
    # Retrieve the focus type for this block
    focus_type = block_focus_map[block]

    # Show block-specific focus instructions
    show_block_focus(win, block, focus_type)

    # Initialize correct answers count for the block
    correct_answers = 0

    # Shuffle trials within each block
    trials = stimuli_combinations * \
        (TRIALS_PER_BLOCK // len(stimuli_combinations))
    random.shuffle(trials)

    for trial_index, trial in enumerate(trials):
        # Determine visual and audio conditions
        visual_high = trial["visual_high"]
        audio_high = trial["audio_high"]

        # Set visual flicker frequencies
        visual_freq_right = VISUAL_FREQ_HIGH if visual_high == "right" else VISUAL_FREQ_LOW
        visual_freq_left = VISUAL_FREQ_LOW if visual_high == "right" else VISUAL_FREQ_HIGH

        # Select appropriate audio combination
        if audio_high == "right":
            audio_right, audio_left = stereo_combinations["right_high_left_low"]
        else:
            audio_right, audio_left = stereo_combinations["right_low_left_high"]

        # Create stereo audio wave
        stereo_wave = np.column_stack(
            (audio_left, audio_right)).astype(np.float32)

        # Display block and trial information
        block_trial_text = visual.TextStim(
            win, text=f"Block: {block}/{NUM_BLOCKS}, Trial: {trial_index + 1}/{TRIALS_PER_BLOCK}", color="white"
        )
        block_trial_text.draw()
        win.flip()
        core.wait(2)  # Show for 1 second

        # Send trial start trigger
        trigger_sender.send_trigger(20)  # trial_start

        # Start non-blocking audio stream
        stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=2)
        stream.start()

        # Calculate frames per cycle for left and right gratings
        frames_per_cycle_left = int(REFRESH_RATE / visual_freq_left)
        frames_per_cycle_right = int(REFRESH_RATE / visual_freq_right)
        on_frames_left = frames_per_cycle_left // 2
        off_frames_left = frames_per_cycle_left - on_frames_left
        on_frames_right = frames_per_cycle_right // 2
        off_frames_right = frames_per_cycle_right - on_frames_right

        # Initialize frame counters and states
        frame_n = 0
        left_on = True
        right_on = True
        last_left_frame = 0
        last_right_frame = 0

        clock = core.Clock()
        response = None
        audio_index = 0  # Starting index for audio playback
        buffer_size = 1024  # Buffer size for audio playback

        # Send trigger for visual stimulation (once at trial start)
        if visual_high == "right":
            # Right visual high or low
            trigger_sender.send_trigger(
                22 if visual_freq_right == VISUAL_FREQ_HIGH else 24)
        else:
            # Left visual high or low
            trigger_sender.send_trigger(
                21 if visual_freq_left == VISUAL_FREQ_HIGH else 23)

        while response is None:
            # Update left grating state
            if left_on and frame_n >= last_left_frame + on_frames_left:
                left_on = False
                last_left_frame = frame_n
            if not left_on and frame_n >= last_left_frame + off_frames_left:
                left_on = True
                last_left_frame = frame_n

            # Update right grating state
            if right_on and frame_n >= last_right_frame + on_frames_right:
                right_on = False
                last_right_frame = frame_n
            if not right_on and frame_n >= last_right_frame + off_frames_right:
                right_on = True
                last_right_frame = frame_n

            # Draw gratings based on current states
            if left_on:
                left_grating.draw()
            if right_on:
                right_grating.draw()

            win.flip()  # Update the window

            # Send trigger for auditory stimulation (once at trial start)
            if audio_high == "right":
                # Right audio high or low
                trigger_sender.send_trigger(
                    26 if audio_right is tone_high else 28)
            else:
                # Left audio high or low
                trigger_sender.send_trigger(
                    25 if audio_left is tone_high else 27)

            # Stream audio dynamically
            if audio_index < len(stereo_wave):
                stream.write(
                    stereo_wave[audio_index: audio_index + buffer_size])
                audio_index += buffer_size
            else:
                audio_index = 0  # Reset audio index to loop sound

            # Check for keyboard responses
            keys = event.getKeys(keyList=["left", "right", "escape"])
            if "escape" in keys:
                stream.stop()
                stream.close()
                win.close()
                core.quit()
            elif keys:
                response = keys[0]

            # Increment frame counter
            frame_n += 1

        # Determine correct response
        correct_response = "right" if visual_high == "right" and audio_high == "right" else "left"

        # Determine if the response is correct
        if response == correct_response:
            correct_answers += 1  # Increment correct answer count

        # Print trial details
        print(f"Trial {trial_index + 1} (Block {block + 1}):")
        print(f"  Visual High: {visual_high}, Audio High: {audio_high}")
        print(
            f"  Correct Response: {correct_response}, User Response: {response}\n")

    # Store block score
    block_scores.append(correct_answers)

    # Inter-block break
    if block < NUM_BLOCKS - 1:
        break_text = visual.TextStim(
            win, text=f"Block {block + 1} complete. Take a 1-minute break.", color="white", pos=(0, 0)
        )
        break_text.draw()
        win.flip()
        core.wait(2)  # Show completion text for 2 seconds
        show_break_with_progress(
            win, BREAK_TIME, correct_answers, TRIALS_PER_BLOCK)

trigger_sender.send_trigger(255)  # session_end

# End of experiment
# Calculate total score
total_score = sum(block_scores)

# Display scores for each block and total score
score_text = "\n".join(
    [f"Block {i + 1}: {score}/{TRIALS_PER_BLOCK}" for i, score in enumerate(block_scores)])
final_score_text = f"Total Score: {total_score}/{NUM_BLOCKS * TRIALS_PER_BLOCK}\n\n" + score_text

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
    win, trigger_sender, duration=3, eyes_open=True)
resting_state_eyes_open.start()  # Eye open phase

resting_state_eyes_closed = RestingState(
    win, trigger_sender, duration=3, eyes_open=False)
resting_state_eyes_closed.start()  # Eye closed phase

transition_text = visual.TextStim(
    win, text="Resting state phase is complete.\n\n Goodbye.", color="white", pos=(0, 0))
transition_text.draw()
win.flip()
core.wait(5)
win.close()
