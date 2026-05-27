import jiwer


class Evalation:
    def __init__(self):
        self.ortho_rules = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToListOfListOfWords(),
        ])

    def wer(self, reference, hypothesis):
        return jiwer.wer(reference, hypothesis)

    def cer(self, reference, hypothesis):
        return jiwer.cer(reference, hypothesis)
