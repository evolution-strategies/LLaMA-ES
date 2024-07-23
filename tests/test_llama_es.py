import unittest
from src.llama_es import tune_cma_es, extract_params_from_llm_answer
import numpy as np
import cma

class TestLlamaEs(unittest.TestCase):

    def test_extract_params_from_llm_answer(self):
        analysis = "CMA_rankmu: 0.3, CMA_rankone: 0.5"
        params = extract_params_from_llm_answer(analysis)
        self.assertEqual(params['CMA_rankmu'], 0.3)
        self.assertEqual(params['CMA_rankone'], 0.5)

    def test_tune_cma_es(self):
        x0 = np.ones(100) * 2
        sigma0 = 1.0
        initial_params = {'CMA_rankmu': 0.1, 'CMA_rankone': 0.1}
        func = cma.ff.hyperelli
        results = tune_cma_es(x0, sigma0, func, initial_params, max_iterations=10)
        self.assertTrue(len(results) > 0)

if __name__ == '__main__':
    unittest.main()