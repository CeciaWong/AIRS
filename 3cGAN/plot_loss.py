import numpy as np
import matplotlib.pyplot as plt
import time

def draw_curve(file_name, setting, Dloss,Gloss,adv,cycle,cycle0,mcycle,slist):
    x = list(range(len(Dloss)))
    plt.figure(figsize=(27*1.4*0.8,4)) #
    plt.suptitle(file_name.strip('.log')+"\n"+setting,fontsize=10)
    ax1=plt.subplot(171)
    plt.plot(x, Dloss, color = "blue", label = "D loss")
    plt.plot(x, Gloss, color = "purple", label = "G loss")
    plt.plot(x, adv, color = "hotpink", label = "adv")
    plt.plot(x, cycle, color = "darkorange", label = "cycle")
    plt.plot(x, cycle0, color = "orange", label = "1-cycle")
    plt.plot(x, mcycle, color = "green", label = "n-cycle")
    plt.legend(loc = 1)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    ax2=plt.subplot(172)
    sslist = []
    for xx in range(24):
        sslist.append(list(np.asarray(slist)[:,xx]))
    num = 0
    plt.plot(x, sslist[0], color = "blue", label = "D loss")
    plt.plot(x, sslist[1], color = "purple", label = "G loss")
    plt.plot(x, sslist[2], color = "hotpink", label = "adv")
    plt.plot(x, sslist[3], color = "darkorange", label = "cycle")
    plt.legend(loc = 1)
    ax3=plt.subplot(173)
    num+=4
    plt.plot(x, sslist[0+num], color = "blue", label = "D loss")
    plt.plot(x, sslist[1+num], color = "purple", label = "G loss")
    plt.plot(x, sslist[2+num], color = "hotpink", label = "adv")
    plt.plot(x, sslist[3+num], color = "darkorange", label = "cycle")
    plt.legend(loc = 1)
    ax4=plt.subplot(174)
    num+=4
    plt.plot(x, sslist[0+num], color = "blue", label = "D loss")
    plt.plot(x, sslist[1+num], color = "purple", label = "G loss")
    plt.plot(x, sslist[2+num], color = "hotpink", label = "adv")
    plt.plot(x, sslist[3+num], color = "darkorange", label = "cycle")
    plt.legend(loc = 1)
    ax5=plt.subplot(175)
    num+=4
    plt.plot(x, sslist[0+num], color = "blue", label = "D loss")
    plt.plot(x, sslist[1+num], color = "purple", label = "G loss")
    plt.plot(x, sslist[2+num], color = "hotpink", label = "adv")
    plt.plot(x, sslist[3+num], color = "darkorange", label = "cycle")
    plt.legend(loc = 1)
    ax6=plt.subplot(176)
    num+=4
    plt.plot(x, sslist[0+num], color = "blue", label = "D loss")
    plt.plot(x, sslist[1+num], color = "purple", label = "G loss")
    plt.plot(x, sslist[2+num], color = "hotpink", label = "adv")
    plt.plot(x, sslist[3+num], color = "darkorange", label = "cycle")
    plt.legend(loc = 1)
    ax7=plt.subplot(177)
    num+=4
    plt.plot(x, sslist[0+num], color = "blue", label = "D loss")
    plt.plot(x, sslist[1+num], color = "purple", label = "G loss")
    plt.plot(x, sslist[2+num], color = "hotpink", label = "adv")
    plt.plot(x, sslist[3+num], color = "darkorange", label = "cycle")

    plt.xticks(range(0,len(Dloss),10))
    plt.legend(loc = 1)
    plt.savefig(file_name.strip('.log')+".png",bbox_inches='tight')
    plt.show()
    time.sleep(2)
    plt.close()
    return None

def read_log(file_name, isval):
    Dloss,Gloss,adv,cycle,slist,cycle0,mcycle = [],[],[],[],[],[],[]
    dloss,gloss,advloss,cycleloss,slosslist,cycle0loss,mcycleloss = [],[],[],[],[],[],[]
    with open(file_name) as f:
        epoch = 1
        while True:
            lines = f.readline()
            if lines.startswith("Namespace"):
                setting = lines.strip("Namespace(").strip(")")#lines.split(",",3)[3].strip(")")
                ismerged = True if int(lines.split("ismerged=")[1].split(",")[0]) else False
            if not lines:
                if len(dloss)>0:
                    Dloss.append(np.mean(dloss))
                    Gloss.append(np.mean(gloss))
                    adv.append(np.mean(advloss))
                    cycle.append(np.mean(cycleloss))
                    cycle0.append(np.mean(cycle0loss))
                    mcycle.append(np.mean(mcycleloss))
                    slist.append(list(np.mean(np.asarray(slosslist),axis=0)))
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
                cycle0.append(np.mean(cycle0loss))
                mcycle.append(np.mean(mcycleloss))
                slist.append(list(np.mean(np.asarray(slosslist),axis=0)))
                dloss,gloss,advloss,cycleloss,slosslist,cycleloss,mcycleloss = [],[],[],[],[],[],[]
            #print(lines)
            lines = lines.split("]                         [")[0]+'] ['+lines.split("]                         [")[1]
            #print(lines)
            dloss.append(float(lines.split("] [")[2].lstrip("D loss: ")))
            gloss.append(float(lines.split("] [")[3].split(", ")[0].lstrip("G loss: ")))
            advloss.append(float(lines.split("] [")[3].split(", ")[1].lstrip("adv: ")))
            cycleloss.append(float(lines.split("] [")[3].split(", ")[2].lstrip("cycle: ")))
            cycle0loss.append(float(lines.split("] [")[3].split(", ")[3].lstrip("cycle0: ")))
            mcycleloss.append(float(lines.split("] [")[3].split(", ")[4].lstrip("mcycle: ").split("]")[0]))
            
            templist = []
            for ii in range(6):
                for jj in range(4):
                    if ii==5 and jj==3:
                        break
                    templist.append(float(lines.split("] [")[ii+4].split(", ")[jj]))
            templist.append(float(lines.split("] [")[9].split(", ")[3].split("]")[0]))
            slosslist.append(templist)

        return setting, Dloss,Gloss,adv,cycle,cycle0,mcycle,slist

if __name__ == '__main__':
    file_name = "3cGAN_AB.log"
    isval = True if file_name.endswith("_val.log") else False
    setting, Dloss,Gloss,adv,cycle,cycle0,mcycle,slist=read_log(file_name,isval)
    draw_curve(file_name, setting, Dloss,Gloss,adv,cycle,cycle0,mcycle,slist)
