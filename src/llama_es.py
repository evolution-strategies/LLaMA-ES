import cma
import numpy as np
import ollama
import re

# Function to tune the CMA-ES algorithm
def tune_cma_es(x0, sigma0, func, initial_params, max_iterations=1000):
    results = []
    params = initial_params

    gen = 0
    while gen < 10:
        # Run CMA-ES with current parameters
        xopt, es = cma.fmin2(func, x0, sigma0, {
            **params, 'maxiter': max_iterations, 'tolfun': 1e-100, 
            'tolfunhist': 1e-100, 'tolx': 1e-100 
        })
        fitness = es.result.fbest

        # Log the results
        results.append({
            'generation': gen + 1,
            'CMA_rankmu': params['CMA_rankmu'],
            'CMA_rankone': params['CMA_rankone'],
            'fitness': fitness,
        })

        # Prepare the prompt for the language model
        history = "\n".join([
            f"Run {r['generation']}: CMA_rankmu={r['CMA_rankmu']}, CMA_rankone={r['CMA_rankone']}, fitness={r['fitness']}" 
            for r in results
        ])
        prompt = (
            f"Given the following history of CMA-ES runs:\n{history}\n\n"
            "Suggest new values for CMA_rankmu and CMA_rankone between 0.0 and 1.0 to minimize the fitness. "
            "Print them in the format 'CMA_rankmu: x, CMA_rankone: y'."
        )

        # Get new parameters from the language model
        llm_answer = ollama.chat(model='llama3:70b', messages=[{'role': 'user', 'content': prompt}])
        analysis = llm_answer['message']['content']
        print("answer using Llama3:70b", analysis)
        
        try:
            new_params = extract_params_from_llm_answer(analysis)
            # Update the parameters for the next iteration
            params['CMA_rankmu'] = new_params['CMA_rankmu']
            params['CMA_rankone'] = new_params['CMA_rankone']
            gen += 1  # Count this as a successful iteration
        except Exception as e:
            print(f"Failed to extract parameters: {e}. Continuing without counting this iteration.")
    
    return results

# Function to extract parameters from the language model's response
def extract_params_from_llm_answer(analysis):
    # Find all mentions of CMA_rankmu and CMA_rankone and their corresponding values
    cma_rankmu_matches = re.findall(r'CMA_rankmu[:\s]*([\d\.]+)', analysis)
    cma_rankone_matches = re.findall(r'CMA_rankone[:\s]*([\d\.]+)', analysis)

    # Check if matches are found and take the last one
    if cma_rankmu_matches:
        cma_rankmu_value = float(cma_rankmu_matches[-1])
    else:
        raise ValueError("CMA_rankmu value could not be found.")

    if cma_rankone_matches:
        cma_rankone_value = float(cma_rankone_matches[-1])
    else:
        raise ValueError("CMA_rankone value could not be found.")

    return {'CMA_rankmu': cma_rankmu_value, 'CMA_rankone': cma_rankone_value}

# Initial parameters and settings
x0 = np.ones(100) * 2
sigma0 = 1.0
initial_params = {'CMA_rankmu': 0.1, 'CMA_rankone': 0.1}
functions = [cma.ff.hyperelli]

# Run the experiments
for func in functions:
    results = tune_cma_es(x0, sigma0, func, initial_params, max_iterations=1000)
    for result in results:
        print(result)