from tqdm import tqdm
from transformers.data.processors.utils import DataProcessor
from seq2seq_data_utils import read_json



class YNAT_Example(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the input sequence.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class YNAT_Processor(DataProcessor):
    """
    Processor for the YNAT(KLUE) data set.
    """

    def get_seq2seq_examples(self, filename, set_type="train"):
        examples = self.get_examples(filename, set_type)

        src_texts = []
        tgt_texts = []
        for e in tqdm(examples, desc="#####\t Get source and target texts ... "):
            src_texts.append("YNAT sentence: " + e.text)
            tgt_texts.append(e.label)

        return (src_texts, tgt_texts)


    def get_examples(self, filename, set_type="train"):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """

        # lines = self._read_tsv(filename)    # read_tsv(filename)
        print(f"#####\t Reading an input file ...\t {filename}")
        lines = read_json(filename)
        examples = self.create_examples(lines, set_type)

        return examples


    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines), desc="#####\t Create examples ... "):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text = line["title"]
            label = line["label"]
            examples.append(
                YNAT_Example(guid=guid, text=text, label=label))
    
        return examples



