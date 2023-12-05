import nltk
from nltk.tag import tnt
from nltk.corpus import indian

# Download the Indian language POS tagged corpus
nltk.download('indian')

# Load the Hindi POS tagged corpus
tagged_sentences = indian.tagged_sents('marathi.pos')

# Train the tagger using the Indian language corpus
tnt_pos_tagger = tnt.TnT()
tnt_pos_tagger.train(tagged_sentences)


# Function to tag tokens in Hindi
def hindi_pos_tag(tokens):
    return tnt_pos_tagger.tag(tokens)


# Function to read and tag tokens from a CoNLL format file and write to another file
def tag_conll_file(file_path, output_file):
    with open(file_path, 'r', encoding='utf-8') as file:
        with open(output_file, 'w', encoding='utf-8') as output:
            for line in file:
                line = line.strip()
                if not line:  # Check for blank lines
                    output.write('\n')
                    continue

                # parts = line.split('\t')
                else:
                    word = line
                    tagged_word = hindi_pos_tag([word])
                    output.write(f"{word}\t{tagged_word[0][1]}\n")
            output.write('\n')  # Ensure a blank line at the end if the input file ends without one


# Path to your CoNLL format file
input_file_path = '/root/home_cdac/ICON/data/Test-Blind/Marathi/marathi_test_blind.tsv'
# Path to the output file
output_file_path = '/root/home_cdac/ICON/data/Test-Blind/Marathi/marathi_test_blind_postagged.tsv'

# Tag tokens in the CoNLL format file and write to the output file
tag_conll_file(input_file_path, output_file_path)
