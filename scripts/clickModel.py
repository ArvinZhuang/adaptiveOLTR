import numpy as np
class clickModel:
    def __init__(self,query_pos_docids, model_type):
        self.query_pos_docids = query_pos_docids
        if model_type == "perfect":
            self.pcR = 1.0
            self.pcNR = 0.0
            self.psR = 0.0
            self.psNR = 0.0
        if model_type == "navigational":
            self.pcR = 0.95
            self.pcNR = 0.05
            self.psR = 0.9
            self.psNR = 0.2
        if model_type == "informational":
            self.pcR = 0.9
            self.pcNR = 0.4
            self.psR = 0.5
            self.psNR = 0.1

    def simulate(self, query, result_list):
        pos_docids = set(self.query_pos_docids[query])
        clicked_doc = []
        for i in range(0, len(result_list)):
            #if the document is relevent
            if result_list[i] in pos_docids:
                if np.random.rand()<= self.pcR:
                    clicked_doc.append(result_list[i])
                    if np.random.rand() <= self.psR:
                        break
            #if the document is nor relevent
            else:
                if np.random.rand() <= self.pcNR:
                    clicked_doc.append(result_list[i])
                    if np.random.rand() <= self.psNR:
                        break
        return clicked_doc