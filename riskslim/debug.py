"""
This provides MATLAB style debugging using iPython

To use
1. import the function "from debug import ipsh"
2. add isph() right before use

Adapted from the following StackExchange post
http://stackoverflow.com/a/23388116/568249
"""

import inspect
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config import Config

# Configure the prompt so that I know I am in a nested (embedded) shell
cfg = Config()
prompt_config = cfg.PromptManager
prompt_config.in_template = 'N.In <\\#>: '
prompt_config.in2_template = '   .\\D.: '
prompt_config.out_template = 'N.Out<\\#>: '

# Messages displayed when I drop into and exit the shell.
banner_msg = ["** ENTERING NESTED INTERPRETER **",
              "Hit Ctrl-D to exit interpreter and continue program.",
              "Note that if you use %kill_embedded, you can fully deactivate",
              "This embedded instance so it will never turn on again." ]

banner_msg = "\n".join(banner_msg)
exit_msg = '** LEAVING NESTED INTERPRETER **'

# Wrap it in a function for more context:
def ipsh():
    ipshell = InteractiveShellEmbed(config=cfg, banner1=banner_msg, exit_msg=exit_msg)
    frame = inspect.currentframe().f_back
    msg = 'Stopped at {0.f_code.co_filename} at line {0.f_lineno}'.format(frame)
    # Go back one level!
    # This is needed because the call to ipshell is inside the function ipsh()
    ipshell(msg, stack_depth=2)