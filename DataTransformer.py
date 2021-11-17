from torch.utils.data import Dataset
from torchvision import transforms

# Custom Class to load data from local system to Pytorch model
class LoadTerrainDataSet(Dataset):
  def __init__(self, path, transform , IsTrain = True,):
    self.path = path
    self.sample = []
    self.label = []
    self.IsTrain = IsTrain
    self.transform = transform

    if self.IsTrain:
      train_data = pd.read_csv(self.path+'raw_train_data_all_Subjects.csv')
      noOfCols = train_data.shape[1]
      noOfRows = train_data.shape[0]
      train_X = train_data.loc[:,['xA','yA','zA','xG','yG','zG']]
      train_y = train_data.loc[:,'y']

      start_index = 1
      end_index = 48
      image_height = 48
      for indx in range(1,noOfRows-image_height):
        start_index = indx
        self.sample.append(train_X[start_index:end_index+1].values)
        self.label.append(train_y[end_index]) 
        end_index += 1

  def __len__(self):
    return len(self.sample)

  def __getitem__(self, index):
    X,y = self.sample[index], self.label[index]
    r,c = X.shape
    X = self.transform(torch.from_numpy(X.reshape((1,r,c))))
    y = torch.from_numpy(np.asarray(y))
    return X,y