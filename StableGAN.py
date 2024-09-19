import torch
from chgnet.model import CHGNet
from pymatgen.core.structure import Structure
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry

# Step 1: Load CHGNet model for energy prediction
chgnet = CHGNet.load()

# Example generator and discriminator functions (assume these are provided in your code)
def generator():
    # Generate hypothetical materials
    # Replace with your actual generation logic
    generated_structures = [...]  # List of generated pymatgen Structure objects
    return generated_structures

def discriminator(structures):
    # Discriminator logic
    # Replace with your actual discriminator logic
    return torch.tensor([...] * len(structures))  # Some output based on the input structures

# Optimizers
optimizer_G = torch.optim.Adam(generator, lr=1e-4)  # Adjust according to your generator function's needs
optimizer_D = torch.optim.Adam(discriminator, lr=1e-4)  # Adjust according to your discriminator function's needs

# Threshold for filtering based on energy above the hull
hull_threshold = 0.2

# Custom function to filter out structures with high energy above the hull
def filter_by_energy_above_hull(structures, predicted_energies):
    entries = []
    for structure, energy in zip(structures, predicted_energies):
        entry = PDEntry(composition=structure.composition, energy=energy)
        entries.append(entry)

    # Create a phase diagram and filter out structures with high energy above the hull
    phase_diagram = PhaseDiagram(entries)
    filtered_structures = []
    for entry in entries:
        energy_above_hull = phase_diagram.get_e_above_hull(entry)
        if energy_above_hull < hull_threshold:
            filtered_structures.append(entry.structure)
    
    return filtered_structures

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Step 2: Generator produces hypothetical materials
    generated_materials = generator()  # Assume this generates a batch of materials (structures)

    # Step 3: Use CHGNet to predict energies for generated materials
    predicted_energies = []
    for structure in generated_materials:
        energy = chgnet.predict_structure(structure)
        predicted_energies.append(energy)

    # Step 4: Filter generated materials by energy above the hull
    filtered_materials = filter_by_energy_above_hull(generated_materials, predicted_energies)

    # Ensure there are filtered materials to continue training
    if len(filtered_materials) == 0:
        print("No stable materials generated in this epoch. Skipping discriminator training.")
        continue

    # Step 5: Train the Discriminator with the filtered stable materials
    optimizer_D.zero_grad()
    real_loss = discriminator(real_materials)  # Assuming `real_materials` is your dataset of stable materials
    fake_loss = discriminator(filtered_materials)  # Fake (generated) filtered materials
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizer_D.step()

    # Step 6: Train the Generator
    optimizer_G.zero_grad()
    g_loss = -fake_loss  # Adjust based on your GAN loss function
    g_loss.backward()
    optimizer_G.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}')


