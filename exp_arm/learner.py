import torch
import torch.nn as nn
import torch.autograd.functional as F
from torch.optim import Adam
from torch.nn.functional import mse_loss as mse
from tqdm import tqdm
from tqdm import trange



DTYPE = torch.float64
torch.set_default_dtype(DTYPE)



class Learner():
    def __init__(self, 
                critic,
                sobolev_order,
                validation_data,
                early_stop_v=1e-5,
                early_stop_vx=1e-4,
                n_epochs=50000,
                lr=1e-3,
                create_anchor=True,
                savePath="./"):
        

        
        self.critic                      =   critic
        self.sobolev_order               =   sobolev_order
        


        self.early_stop_v                =  early_stop_v
        self.early_stop_vx               =  early_stop_vx
        self.n_epochs                    =  n_epochs
        self.lr                          =  lr
        self.create_anchor               =  create_anchor

       
        
        self.optimizer                   =   Adam(critic.parameters(), lr = self.lr)


        ### Validation  dataloader
        self.validation_x0s              =   validation_data[0]
        self.validation_targetv          =   validation_data[1]
        

        self.savePath                    =   savePath

    def learn(self,training_data,episode=0,name_prefix=""):


        training_data   =   training_data
        dataset_size    =   training_data.dataset.tensors[0].shape[0]

        self.progress_bar                =   trange(self.n_epochs, desc = " Training ...")




        critic  = self.critic
       
        
        
        
        training_loss_vx             =   []
        training_loss_v              =   []

        
        
        
        validation_loss_v            =   []

        
        
        
        for epoch,_ in zip(range(self.n_epochs), self.progress_bar):

            ##  Set agent to train mode
            critic.train()

            loss_v, loss_vx     =   0.0, 0.0


            for idx, (data) in enumerate(training_data):

                x0s, *target        =   data


                predictions         =   list(critic(x0s, order = 1))

                losses              =   list(map(mse, predictions, target))
                self.optimizer.zero_grad()
                for loss in losses[:self.sobolev_order+1]:
                    loss.backward(retain_graph=True)


                self.optimizer.step()

                loss_v  +=  x0s.shape[0] * losses[0].item()
                loss_vx +=  x0s.shape[0] * losses[1].item()

            loss_v  =   loss_v      /   dataset_size
            loss_vx =   loss_vx     /   dataset_size

            training_loss_v.append(loss_v)
            training_loss_vx.append(loss_vx)


            validation_loss    =   self.validate()
            validation_loss_v.append(validation_loss)


            if epoch % 10 == 0:
                lv  =   "{:.1e}".format(loss_v)
                lvx =   "{:.1e}".format(loss_vx)
                vv  =   "{:.1e}".format(validation_loss)

                desc = f"Epoch:{epoch}::: Training:--> Value Loss:{lv}, Gradient Loss:{lvx}, Testing:--> Value Loss:{vv}"

                self.progress_bar.set_description(desc)
                self.progress_bar.refresh()
            
            if (loss_v < self.early_stop_v and loss_vx < self.early_stop_vx and self.sobolev_order == 1):
                    
                print(f"[INFO]Early Stopping criterion met at epoch {epoch} !! Break")
                self.progress_bar
                break

            if (loss_v < self.early_stop_v and self.sobolev_order == 0):
                
                print(f"[INFO]Early Stopping criterion met at epoch {epoch} !! Break")
                self.progress_bar
                break


            del validation_loss,losses,loss_v, loss_vx,predictions



        info_dict = self._logging_info(episode = episode, 
                                    loss_training   = [training_loss_v, training_loss_vx],
                                    loss_validation = [validation_loss_v])
        
        del training_data

        torch.save(critic, self.savePath+name_prefix+'.pth')
        self.critic = critic
        return critic,info_dict



    
    def _logging_info(self,loss_training:list, loss_validation:list, episode = 0):
        info_dict   =   [ 
                        {"Training Loss V":loss_training[0]},
                        {"Training Loss Vx":loss_training[1]},
                        {"Validation Loss V":loss_validation[0]},                                   
                        ] 
        return info_dict

    
    def validate(self) -> float:
        with torch.no_grad():
            predictions         =   self.critic(self.validation_x0s, order = 0)
        
        loss              =   mse(predictions, self.validation_targetv)
        return loss.item()






    