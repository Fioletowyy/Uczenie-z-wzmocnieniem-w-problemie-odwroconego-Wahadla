import gym
import numpy as np
import math
from collections import deque
import matplotlib
import matplotlib.pyplot as plt

def plot_durations(scores):    #wyswietlanie wykresu
        plt.figure(2)
        plt.clf()

        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(scores)

  

        plt.pause(0.001)  
        if is_ipython:
            display.clear_output(wait=True)
            display.display(plt.gcf())





def dyskretyzacja(obserwacja,env,wymiary):
            zakres_gorny = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(220)]  #określenie zakreśów przedziałów
            zakres_dolny = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(220)]
            n_obserwacja =[int(round((obserwacja[i]-zakres_dolny[i]) / (zakres_gorny[i] - zakres_dolny[i])*(wymiary[i]-1)))for i in range(len(obserwacja))]    # przeskalowanie
            return tuple(n_obserwacja)



       
            






def gra(env,Q_macierz,wymiar,wywolania,czas,min_alfa,min_epsilon,gamma,nauka):
        punkty = deque(maxlen=2000)
        

        for k in range(nauka):         #nauczanie
      
            s = dyskretyzacja(env.reset(),env,wymiar)
        
            alfa = max(min_alfa, min(1.0, 1.0 - math.log10((k + 1) /20)))
            if k<350:
                epsilon = max(min_epsilon, min(1, 1.0 - math.log10((k + 1) /35)))
            else:
                epsilon=0
            
            d = False
            i = 0

            while not d:
          
               
                if (np.random.random() <= epsilon):          # omijanie lokalnych minimów
                   a=env.action_space.sample() 
                else: 
                   a=np.argmax(Q_macierz[s])
           
                obserwacja, r, d, _ = env.step(a)
                s1 = dyskretyzacja(obserwacja,env,wymiar)
                if (abs(obserwacja[0])>0.8) :         # funkcje kar      
                    r=-10                   
                if (d and (i<199)):
                     r=- 60
                if k<nauka-nauka/4:              # ucz przez 3/4 wywolania 1/4 reprezentuje na wykresie nauczony algorytm
                    Q_macierz[s][a] += alfa * (r + gamma * np.max(Q_macierz[s1]) - Q_macierz[s][a])
                s= s1
                i += 1
            punkty.append(i)
        plot_durations(punkty)


        for j in range(wywolania):                       #algorytm demonstracja 
        
            s = dyskretyzacja(env.reset(),env,wymiar)
        

            d = False
            i = 0

            while not d:
                env.render()
               
                if (np.random.random() <= epsilon):
                   a=env.action_space.sample() 
                else: 
                   a=np.argmax(Q_macierz[s])
           
                obserwacja, r, d, _ = env.step(a)
                s1 = dyskretyzacja(obserwacja,env,wymiar)
               
               
                s= s1
                i += 1
            
           
        
            
          
  

if __name__ == "__main__":
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:                                 #inicjalizacja odswiezania wyswietlania
        from IPython import display

    plt.ion()

    wymiar=(1, 1, 65, 42,)  # możliwe wykorzystanie pozostałych zmiennych stanu
    wywolania=5
    czas=199
    min_alfa=0.01
    min_epsilon=0.1 
    gamma=1
  
    nauka=600
    env = gym.make('CartPole-v0')
    Q_macierz= np.zeros(wymiar + (env.action_space.n,))    #inicjalizacja wymiarów zerowej macierzy Q/


    gra(env,Q_macierz,wymiar,wywolania,czas,min_alfa,min_epsilon,gamma,nauka)
    plt.ioff()
    plt.show()
