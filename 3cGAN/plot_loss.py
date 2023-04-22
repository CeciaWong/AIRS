import numpy as np
import matplotlib.pyplot as plt
import time

def draw_curve(file_name, setting, Dloss,Gloss,adv,cycle,isval):
    x = list(range(len(Dloss)))
    plt.figure(figsize=(27*0.8,4))
    plt.suptitle(file_name.strip('.log')+"\n"+setting,fontsize=10)
    ax1=plt.subplot(151)
    plt.plot(x, Dloss, color = "red", label = "D loss")
    plt.plot(x, Gloss, color = "blue", label = "G loss")
    plt.plot(x, adv, color = "green", label = "adv")
    plt.plot(x, cycle, color = "orange", label = "cycle")
    plt.legend()
    plt.xlabel("epoch")
    savesign = "val" if isval else "train"
    plt.ylabel(savesign+" loss")
    ax2=plt.subplot(152)
    plt.plot(x, Gloss, color = "blue", label = "G loss")
    plt.legend()
    ax3=plt.subplot(153)
    plt.plot(x, Dloss, color = "red", label = "D loss")
    plt.legend()
    ax4=plt.subplot(154)
    plt.plot(x, adv, color = "green", label = "adv")
    plt.legend()
    ax5=plt.subplot(155)
    plt.plot(x, cycle, color = "orange", label = "cycle")
    plt.xticks(range(0,len(Dloss),10))
    plt.legend()
    plt.savefig(file_name.strip('.log')+"_"+savesign+".png",bbox_inches='tight')
    plt.show()
    time.sleep(2)
    plt.close()
    return None

def read_log(file_name, isval):
    Dloss,Gloss,adv,cycle = [],[],[],[]
    dloss,gloss,advloss,cycleloss = [],[],[],[]
    with open(file_name) as f:
        epoch = 1
        while True:
            lines = f.readline()
            if lines.startswith("Namespace"):
                setting = lines.split(",",3)[3].strip(")")
            if not lines:
                if len(dloss)>0:
                    Dloss.append(np.mean(dloss))
                    Gloss.append(np.mean(gloss))
                    adv.append(np.mean(advloss))
                    cycle.append(np.mean(cycleloss))
                break
            if not isval:
                if not lines.startswith("[Epoch"):
                    continue
            if isval:
                if not lines.startswith("*[Epoch"):
                    continue
            epoch_now = int(lines.split("/", 1)[0].lstrip("*").lstrip("[Epoch "))
            if epoch_now != epoch:
                epoch = epoch_now
                Dloss.append(np.mean(dloss))
                Gloss.append(np.mean(gloss))
                adv.append(np.mean(advloss))
                cycle.append(np.mean(cycleloss))
                dloss,gloss,advloss,cycleloss = [],[],[],[]
            dloss.append(float(lines.split("] [")[2].lstrip("D loss: ")))
            gloss.append(float(lines.split("] [")[3].split(", ")[0].lstrip("G loss: ")))
            advloss.append(float(lines.split("] [")[3].split(", ")[1].lstrip("adv: ")))
            cycleloss.append(float(lines.split("] [")[3].split(", ")[2].lstrip("cycle: ").split("]")[0]))
        return setting, Dloss,Gloss,adv,cycle

if __name__ == '__main__':
    file_name = "rb3.log"
    for isval in [False, True]:
        setting, Dloss,Gloss,adv,cycle=read_log(file_name,isval)
        draw_curve(file_name, setting, Dloss,Gloss,adv,cycle,isval)
