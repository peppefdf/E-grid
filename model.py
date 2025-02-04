import gurobipy as gr
from gurobipy import GRB
import numpy as np
from geopy.distance import geodesic

def define_model(input_data):

    num_generators = input_data['num_generators']
    num_users = input_data['num_users']
    num_storage = input_data['num_storage']
    generator_positions = input_data['generator_positions'].reshape((num_generators, 2))
    user_positions = input_data['user_positions'].reshape((num_users, 2))
    storage_positions = input_data['storage_positions'].reshape((num_storage, 2))
    generator_power = input_data['generator_power']
    user_requirements = input_data['user_requirements']
    storage_power_capacity = input_data['storage_power_capacity']
    initial_stored_energy = input_data['stored_energy']
    storage_Emax = input_data['storage_Emax']
    time_power = input_data['time_power']
    resistivity_factor = input_data['resistivity_factor']

    V0 = 230

    # Define the model
    model = gr.Model(GRB.BINARY)

    # Define the variables
    x_GU = model.addVars(num_generators, num_users, lb=0, ub=1, vtype=gr.GRB.BINARY)
    x_GS = model.addVars(num_generators, num_storage, lb=0, ub=1, vtype=gr.GRB.BINARY)
    x_SU = model.addVars(num_storage, num_users, lb=0, ub=1, vtype=gr.GRB.BINARY)

    stored_E = model.addVars(num_storage, lb=0, ub=max(storage_Emax), name='stored_E')
    
    # Define the power loss as dictionaries
    #resistance_GU = {(i, j): resistivity_factor*np.linalg.norm([generator_positions[i, 0] - user_positions[j, 0], generator_positions[i, 1] - user_positions[j, 1]]) for i in range(num_generators) for j in range(num_users)}
    resistance_GU = {(i, j): resistivity_factor*geodesic((generator_positions[i, 0], generator_positions[i, 1]), (user_positions[j, 0], user_positions[j, 1])).meters for i in range(num_generators) for j in range(num_users)}
    
    #resistance_SU = {(i, j): resistivity_factor*np.linalg.norm([storage_positions[i, 0] - user_positions[j, 0], storage_positions[i, 1] - user_positions[j, 1]]) for i in range(num_storage) for j in range(num_users)}
    resistance_SU = {(i, j): resistivity_factor*geodesic((storage_positions[i, 0], storage_positions[i, 1]), (user_positions[j, 0], user_positions[j, 1])).meters for i in range(num_storage) for j in range(num_users)}
    
    #resistance_GS = {(i, j): resistivity_factor*np.linalg.norm([generator_positions[i, 0] - storage_positions[j, 0], generator_positions[i, 1] - storage_positions[j, 1]]) for i in range(num_generators) for j in range(num_storage)}
    resistance_GS = {(i, j): resistivity_factor*geodesic((generator_positions[i, 0], generator_positions[i, 1]), (storage_positions[j, 0], storage_positions[j, 1])).meters for i in range(num_generators) for j in range(num_storage)}

    power_loss_GU = model.addVars(num_generators, num_users, lb=0, ub=1000, name='power_loss_GU')
    power_loss_GS = model.addVars(num_generators, num_storage, lb=0, ub=1000, name='power_loss_GS')
    power_loss_SU = model.addVars(num_storage, num_users, lb=0, ub=1000, name='power_loss_SU')

    # Define the new variables
    power_GU = {}
    power_GS = {}
    power_SU = {}

    for i in range(num_generators):
        for j in range(num_users):
            power_GU[i, j] = model.addVar(lb = 0, ub=generator_power[i], name=f'power_GU_{i}_{j}')      

    for i in range(num_generators):
        for j in range(num_storage):
            power_GS[i, j] = model.addVar(lb = 0, ub=generator_power[i], name=f'power_GS_{i}_{j}')

    for i in range(num_storage):
        for j in range(num_users):
            power_SU[i, j] = model.addVar(lb = 0, ub=storage_power_capacity[i], name=f'power_SU_{i}_{j}')

    for i in range(num_generators):
        model.addConstr(gr.quicksum([power_GU[i, j]* x_GU[i, j] for j in range(num_users)]) + gr.quicksum([power_GS[i, j]* x_GS[i, j] for j in range(num_storage)]) <= generator_power[i], 'generator_limits_{}'.format(i))

    for i in range(num_storage):
        model.addConstr(gr.quicksum([power_SU[i, j]* x_SU[i, j] for j in range(num_users)]) <= storage_power_capacity[i], 'storage_limits_{}'.format(i))


    # Define the constraint
    # Relax the connection constraint
    #for j in range(num_users):
    #    model.addConstr(gr.quicksum([x_GU[i, j] for i in range(num_generators)]) >= 0.5, 'connection_constraint_users_{}'.format(j))
    for j in range(num_users):
        model.addConstr(gr.quicksum([x_GU[i, j] for i in range(num_generators)]) >= 1.0, 'connection_constraint_users_{}'.format(j))


    # Add a constraint to ensure that each storage device is connected to only the nearest generator
    # Calculate the distance between each generator and storage device
    distances = np.linalg.norm(generator_positions[:, np.newaxis] - storage_positions, axis=2)
    # Find the index of the nearest generator for each storage device
    nearest_generators = np.argmin(distances, axis=0)
    for i in range(num_storage):
        model.addConstr(gr.quicksum([x_GS[j, i] for j in range(num_generators)]) == 1, f'storage_connection_constraint_{i}')
        for j in range(num_generators):
            if j != nearest_generators[i]:
                model.addConstr(x_GS[j, i] == 0, f'storage_connection_constraint_{i}_{j}')


    for i in range(num_generators):
        for j in range(num_users):
            model.addConstr(power_loss_GU[i, j] == 0.5*(V0 - (V0**2 - 4 * resistance_GU[i,j]*user_requirements[j])**0.5), 'power_loss_GU_{}'.format(i, j))
    for i in range(num_storage):
        for j in range(num_users):
            model.addConstr(power_loss_SU[i, j] == 0.5*(V0 - (V0**2 - 4 * resistance_SU[i,j]*user_requirements[j])**0.5), 'power_loss_SU_{}'.format(i, j))
    for i in range(num_generators):
        for j in range(num_storage):
            model.addConstr(power_loss_GS[i, j] == 0.5*(V0 - (V0**2 - 4 * resistance_GS[i,j]*storage_power_capacity[j])**0.5), 'power_loss_SU_{}'.format(i, j))


    #for i in range(num_generators):
    #    for j in range(num_users):
    #        model.addConstr(power_loss_GU[i, j] == resistance_GU[i, j]*(user_requirements[j]/230)**2, 'power_loss_GU_{}'.format(i, j))
    #for i in range(num_storage):
    #    for j in range(num_users):
    #        model.addConstr(power_loss_SU[i, j] == resistance_SU[i, j]*(user_requirements[j]/230)**2, 'power_loss_SU_{}'.format(i, j))
    #for i in range(num_generators):
    #    for j in range(num_storage):
    #        model.addConstr(power_loss_GS[i, j] == resistance_GS[i, j]*(storage_power_capacity[j]/230)**2, 'power_loss_GS_{}'.format(i, j)) 


    for j in range(num_users):
        model.addConstr(
            gr.quicksum([power_GU[i, j] * x_GU[i, j] - power_loss_GU[(i, j)]* x_GU[i, j] for i in range(num_generators)]) +
            gr.quicksum([power_SU[i, j] * x_SU[i, j] - power_loss_SU[(i, j)]* x_SU[i, j] for i in range(num_storage)]) ==
            user_requirements[j],'user_energy_balance_{}'.format(j)
        )

    # Energy balance on storage devices: the accumulated energy should not exceed the maximum capacity:
    # (P_in*t - P_out*t) < = E_max - E_initial
    for j in range(num_storage):
        model.addConstr(initial_stored_energy[j] +
            gr.quicksum([power_GS[i, j] * x_GS[i, j] for i in range(num_generators)]) * time_power -
            gr.quicksum([power_loss_GS[i, j] * x_GS[i, j] for i in range(num_generators)]) * time_power -
            gr.quicksum([power_SU[j, k] * x_SU[j, k] for k in range(num_users)]) * time_power <=
            storage_Emax[j],'stored_energy_balance_max_{}'.format(j)
        )         

    # Energy balance on storage devices: the stored energy - out flow should not be negative:
    # stored_energy + P_in*t - P_out*t >= 0
    for j in range(num_storage):
        model.addConstr(initial_stored_energy[j] +
            gr.quicksum([power_GS[i, j] * x_GS[i, j] for i in range(num_generators)]) * time_power -
            gr.quicksum([power_loss_GS[i, j] * x_GS[i, j] for i in range(num_generators)]) * time_power -
            gr.quicksum([power_SU[j, k] * x_SU[j, k] for k in range(num_users)]) * time_power >=
            0,'stored_energy_balance_min_{}'.format(j)
        )

    for j in range(num_storage):
        model.addConstr(initial_stored_energy[j] +
            gr.quicksum([power_GS[i, j] * x_GS[i, j] for i in range(num_generators)]) * time_power -
            gr.quicksum([power_loss_GS[i, j] * x_GS[i, j] for i in range(num_generators)]) * time_power -
            gr.quicksum([power_SU[j, k] * x_SU[j, k] for k in range(num_users)]) * time_power ==
            stored_E[j],'stored_Energy_{}'.format(j)
        ) 

    #obj = gr.quicksum([generator_power[i] for i in range(num_generators)]) - \
    #    gr.quicksum([power_GU[i, j] * x_GU[i, j] for i in range(num_generators) for j in range(num_users)]) - \
    #    gr.quicksum([power_GS[i, j] * x_GS[i, j] for i in range(num_generators) for j in range(num_storage)])

    #obj = gr.quicksum([generator_power[i] for i in range(num_generators)]) - \
    #    gr.quicksum([power_GU[i, j] * x_GU[i, j] for i in range(num_generators) for j in range(num_users)]) - \
    #    gr.quicksum([power_GS[i, j] * x_GS[i, j] for i in range(num_generators) for j in range(num_storage)]) + \
    #    0.0001 * gr.quicksum([storage_Emax[j] for j in range(num_storage)])

    obj = gr.quicksum([generator_power[i] for i in range(num_generators)]) - \
        gr.quicksum([power_GU[i, j] * x_GU[i, j] for i in range(num_generators) for j in range(num_users)]) - \
        gr.quicksum([power_GS[i, j] * x_GS[i, j] for i in range(num_generators) for j in range(num_storage)]) + \
        0.01 * gr.quicksum([power_loss_GU[i, j]*x_GU[i, j] for i in range(num_generators) for j in range(num_users)]) + \
        0.01 * gr.quicksum([power_loss_GS[i, j]*x_GS[i, j] for i in range(num_generators) for j in range(num_storage)]) + \
        0.01 * gr.quicksum([power_loss_SU[i, j]*x_SU[i, j] for i in range(num_storage) for j in range(num_users)])

    # Add regularization to the objective function
    #obj += 0.01 * gr.quicksum([power_GU[i, j]**2 for i in range(num_generators) for j in range(num_users)])

    model.setObjective(obj, gr.GRB.MINIMIZE)

    return model, x_GU, x_GS, x_SU, power_GU, power_GS, power_SU, power_loss_GU, power_loss_GS, power_loss_SU, stored_E