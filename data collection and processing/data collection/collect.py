import random
import os, sys
import csv
from psychopy import visual, core, sound, event, logging as plog

# =======================
# Suppress PsychoPy Messages & Frame-rate Prompt
# =======================
plog.console.setLevel(plog.CRITICAL)
_devnull = open(os.devnull, 'w')
_old_stdout, _old_stderr = sys.stdout, sys.stderr
sys.stdout, sys.stderr = _devnull, _devnull

# =======================
# Window & Stimuli Setup
# =======================
win = visual.Window(fullscr=True, color='black', units='height', autoLog=False)
sys.stdout, sys.stderr = _old_stdout, _old_stderr
_devnull.close()

# Fixation cross and arrow cues
cross = visual.TextStim(win, text='+', color='white', height=0.1)
left_arrow = visual.ImageStim(win, image='left_arrow.png', size=0.2)
right_arrow = visual.ImageStim(win, image='right_arrow.png', size=0.2)
arrow_half = 0.2 / 2
cross_half = 0.1 / 2
cross_offset_y = arrow_half + cross_half + 0.02

# Sound cue
beep = sound.Sound('beep.wav')

# Exit helper
def check_exit():
    if event.getKeys(keyList=['escape']):
        win.close()
        core.quit()

# =======================
# Wait for 'S' key to start
# =======================
start_msg = visual.TextStim(win,
    text="Press 'S' to start the experiment or 'ESC' to exit.",
    color='white', height=0.07
)
start_msg.draw()
win.flip()
start_keys = event.waitKeys(keyList=['s', 'escape'])
if 'escape' in start_keys:
    win.close()
    core.quit()

# =======================
# Experiment Settings (Single Subject & Session)
# =======================
current_subject = 'A03'  # Change as needed
current_session = 'T'    # 'T' for Training or 'E' for Evaluation
n_runs = 6
pre_stim = 2.0
stim_duration = 1.0
imagine_duration = 4.0
rest_duration = 3      # Rest now 3 seconds

# Prepare log file for labeling
log_filename = f"{current_subject}_{current_session}_log.csv"
log_file = open(log_filename, 'w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['subject', 'session', 'run', 'trial_index', 'trial_type'])

print(f"Starting subject {current_subject} session {current_session}")

# =======================
# Trial Loop
# =======================
for run in range(1, n_runs + 1):
    # Randomized, balanced trials
    trials = [1]*20 + [2]*20
    random.shuffle(trials)

    for idx, trial_type in enumerate(trials, start=1):
        # Log trial info
        log_writer.writerow([current_subject, current_session, run, idx,
                             'left' if trial_type == 1 else 'right'])
        log_file.flush()

        t_clock = core.Clock()
        # --- Fixation + beep ---
        cross.setPos((0, 0)); cross.draw(); win.flip(); beep.play(); core.wait(pre_stim); check_exit()

        # --- Arrow cue + cross above ---
        (left_arrow if trial_type == 1 else right_arrow).draw()
        cross.setPos((0, cross_offset_y)); cross.draw()
        win.flip(); core.wait(stim_duration); check_exit()

        # --- Imagery until 6s total ---
        while t_clock.getTime() < (pre_stim + stim_duration + imagine_duration):
            cross.setPos((0, 0)); cross.draw(); win.flip(); check_exit()

        # --- Rest with countdown ---
        for sec in range(rest_duration, 0, -1):
            count_txt = visual.TextStim(win, text=str(sec), color='white', height=0.1)
            count_txt.draw()
            win.flip()
            core.wait(1)
            check_exit()

    # --- Break between runs ---
    break_msg = visual.TextStim(
        win,
        text=f'Run {run} complete. Press SPACE to continue or ESC to exit.',
        color='white', height=0.05
    )
    break_msg.draw(); win.flip()
    keys = event.waitKeys(keyList=['space','escape'])
    if 'escape' in keys:
        break

# =======================
# Cleanup
# =======================
log_file.close()
win.close()
core.quit()
