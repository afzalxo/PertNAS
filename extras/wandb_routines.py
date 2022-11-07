import numpy as np
from genotypes import PRIMITIVES
import wandb

def _switches_to_bool(switches):
    k = len(switches)
    b_sw = np.zeros((k, len(PRIMITIVES)))
    for i in range(len(switches)):
        for j in range(len(switches[i])):
            if switches[i][j] != 100:
                b_sw[i][switches[i][j]] = True
    return b_sw

def _log_switch_table(switches, note):
    k = len(switches)
    col = [PRIMITIVES[i] for i in range(len(PRIMITIVES))]
    b_sw = _switches_to_bool(switches)
    wandb.log({'switch-vals-'+str(note): wandb.Table(data=b_sw, columns=col)}, commit=False)
    wandb.log({'switch-heatmap-'+str(note): wandb.plots.HeatMap(col, range(0, k), b_sw)}, commit=False)

def _log_imp_table(importance, switches, edge, note, full=False):
    k = len(switches)
    if not full:
        imp = -10*np.ones((0, len(PRIMITIVES)))
        #imp = np.full((14, len(PRIMITIVES)), None)
        for j in range(len(switches[edge])):
            if switches[edge][j] != 100:
                imp[0][switches[edge][j]] = importance[j]
        col = [PRIMITIVES[i] for i in range(len(PRIMITIVES))]

        wandb.log({'importance-vals-'+str(note): wandb.Table(data=imp, columns=col)}, commit=False)
        wandb.log({'importance-heatmap-'+str(note): wandb.plots.HeatMap(col, [edge], imp, show_text=True)}, commit=False)
    else:
        imp = -10*np.ones((k, len(PRIMITIVES)))
        for i in range(len(switches)):
            for j in range(len(switches[i])):
                if switches[i][j] != 100:
                    imp[i][switches[i][j]] = importance[i][j]
        col = [PRIMITIVES[i] for i in range(0, len(PRIMITIVES))]

        wandb.log({'importance-vals-discretize-'+str(note): wandb.Table(data=imp, columns=col)}, commit=False)
        wandb.log({'importance-heatmap-discretize-'+str(note): wandb.plots.HeatMap(col, range(0, k), imp, show_text=True)}, commit=False)

