import nltk

import rnasl.gconst as gc


class PCFG_RNAStructSimple:
    def __init__(self):
        """
        X: some string
        L: loop (with min loop size 4)
        F: free/unpaired base
        T: terminal symbol (paired base)
        TXT: base pair
        """
        pcfg_grammar = """
                    S -> X [0.7] | F [0.1] | F F [0.1] | F F F [0.1]
                    X -> TXT [0.4] | X TXT [0.1] | XF [0.4] | F F F F [0.1]
                    XF -> X F [0.5] | F X [0.5]
                    TXT -> GC [0.65] | AU [0.25] | GU [0.1]
                    GC -> G X C [0.5] | C X G [0.5]
                    AU -> A X U [0.5] | U X A [0.5]
                    GU -> G X U [0.5] | U X G [0.5]
                    F -> A [0.25] | C [0.25] | G [0.25] | U [0.25]
                    A -> 'a' [1.0]
                    C -> 'c' [1.0]
                    G -> 'g' [1.0]
                    U -> 'u' [1.0]
                    """
        self.grammar = nltk.PCFG.fromstring(pcfg_grammar)
        self.grammar.EPSILON = 0.01

    def parse(self, rna_str, get_top_x=1, show_tree=False):
        """Parses the given RNA string and returns a list of vienna strings."""
        if get_top_x == 1:
            self.parser = nltk.ViterbiParser(self.grammar)
        else: # get best 'get_top_x' number of predicted structures
            self.parser = nltk.InsideChartParser(self.grammar)
        to_parse = list(rna_str.lower())
        trees = list(self.parser.parse(to_parse))
        viennas = []
        i = 0
        for tree in trees:
            if i >= get_top_x: break
            if show_tree:
                tree.pretty_print()
            vienna_struct = self._tree_to_vienna(tree)
            viennas.append(vienna_struct)
            i += 1
        return viennas

    def train(self, sequence, structure):
        """
        Problem: we need a sequence + rule elements here, not seq + vienna
        Idea: manually create a function that calculates applied rules for this particular grammar from a vienna string,
              then training the probabilities consists of counting occurrences of each rule
              -> lots of manual work, but maybe good comparison to CRF approach
        """
        return

    def _tree_to_vienna(self, tree):
        """Converts a parse tree for this grammar to an RNA structure in vienna notation"""
        if isinstance(tree, nltk.Tree):
            lbl = tree.label()
            if lbl == "TXT":
                return "(" + "".join(self._tree_to_vienna(child) for child in tree) + ")"
            elif lbl == "F":
                return "."
            else:
                return "".join(self._tree_to_vienna(child) for child in tree)
        else:
            return ""


def predict_structure(rna_str):
    pcfg = PCFG_RNAStructSimple()
    num_predictions = 1
    vienna_structs = pcfg.parse(rna_str, get_top_x=num_predictions)

    if gc.VERBOSE and num_predictions != 1:
        print(rna_str.upper())
        for vienna in vienna_structs:
            print(vienna)

    return vienna_structs[0]
