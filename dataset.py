
from sentence_transformers import InputExample
from torch.utils.data import Dataset


# Class implementation for a triplet data set contain positive and negative examples
class TripletDataset(Dataset):
    # -------------------------------------------------------------------------
    def __init__(self, triplet_list) -> None:
        super().__init__()
        self.triplets = []
        self.build_triplets_input_examples(triplet_list)

    # -------------------------------------------------------------------------
    def build_triplets_input_examples(self, triplet_list):

        for triplet in triplet_list:
            self.triplets.append(InputExample(texts=[triplet[0], triplet[1], triplet[2]]))

    # -------------------------------------------------------------------------
    def __len__(self):
        return (len(self.triplets))

    # -------------------------------------------------------------------------
    def __getitem__(self, index):
        return self.triplets[index]
