import torch
import tqdm
import matplotlib.pyplot as plt
import yaml


def loss_fn(recipe, vectors, targets):
    calories_vector, proteins_vector, fat_vector, carbs_vector, enjoyment_vector = vectors
    calories_target, proteins_target, fat_target, carbs_target = targets

    recipe_calories = recipe @ calories_vector
    recipe_proteins = recipe @ proteins_vector
    recipe_fat = recipe @ fat_vector
    recipe_carbs = recipe @ carbs_vector

    # Try to stick to the target calories and macros.
    loss = (recipe_calories - calories_target) ** 2
    loss += (recipe_fat - fat_target) ** 2
    loss += (recipe_carbs - carbs_target) ** 2

    # Try to reach the target proteins, but more is okay.
    loss += (proteins_target - recipe_proteins).clamp(min=0) ** 2

    # Add a term to promote variety in the choice of foods.
    # Consider the recipe as a distribution over the foods.
    # Then maximize the entropy of the distribution.
    recipe_distribution = torch.nn.functional.softmax(recipe, dim=0) + 1e-6
    recipe_entropy = -(recipe_distribution * torch.log2(recipe_distribution)).sum()
    loss += recipe_entropy * 100

    # Add a term to promote enjoying the food.
    recipe_enjoyment = recipe @ enjoyment_vector
    loss -= recipe_enjoyment

    return loss, recipe_calories, recipe_entropy, recipe_enjoyment


def main():
    # Build the ingredients database.
    with open("foods.yaml") as stream:
        try:
            configuration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    foods = {"names": [], "calories": [], "proteins": [], "fats": [], "carbs": [], "enjoyment": []}
    for food_name in configuration["foods"].keys():
        macros = configuration["foods"][food_name]

        foods["names"].append(food_name)
        foods["calories"].append(macros["calories"])
        foods["proteins"].append(macros["proteins"])
        foods["fats"].append(macros["fats"])
        foods["carbs"].append(macros["carbs"])
        foods["enjoyment"].append(macros["enjoyment"])

    calories_vector = torch.tensor(foods["calories"], dtype=torch.float32) / 100.0
    proteins_vector = torch.tensor(foods["proteins"], dtype=torch.float32) / 100.0
    fat_vector = torch.tensor(foods["fats"], dtype=torch.float32) / 100.0
    carbs_vector = torch.tensor(foods["carbs"], dtype=torch.float32) / 100.0
    enjoyment_vector = torch.tensor(foods["enjoyment"], dtype=torch.float32)

    vectors = (calories_vector, proteins_vector, fat_vector, carbs_vector, enjoyment_vector)
    targets = (configuration["targets"]["calories"], configuration["targets"]["proteins"], configuration["targets"]["fats"], configuration["targets"]["carbs"])

    # Optimize the recipe.
    parameters = torch.nn.Parameter((torch.randn((4,), dtype=torch.float32) + 1) * 100, requires_grad=True)
    optimizer = torch.optim.SGD([parameters], lr=0.01)

    iterations = 5000
    pbar = tqdm.tqdm(range(iterations))

    best_recipe = None
    best_loss = float("inf")
    best_recipe_entropy = float("inf")
    best_recipe_enjoyment = float("inf")

    for i in pbar:
        optimizer.zero_grad()
        loss, calories, recipe_entropy, recipe_enjoyment = loss_fn(torch.abs(parameters), vectors, targets)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_recipe = parameters.clone().detach()
            best_recipe_entropy = recipe_entropy.item()
            best_recipe_enjoyment = recipe_enjoyment.item()

        pbar.set_description(f"Loss: {loss.item():.2f} (calories: {calories.item():.2f})")

    best_recipe = torch.abs(best_recipe)

    print(f"Optimal recipe found:")
    for i in range(len(foods["names"])):
        if best_recipe[i].item() > 0.1:
            print(f" - {foods['names'][i]}: {best_recipe[i].item():.2f}g")

    print(f"Total calories: {best_recipe @ calories_vector:.2f}")
    print(f"Total proteins: {best_recipe @ proteins_vector:.2f}")
    print(f"Total fat: {best_recipe @ fat_vector:.2f}")
    print(f"Total carbs: {best_recipe @ carbs_vector:.2f}")
    print(f"Loss (lower is better): {best_loss}")
    print(f"Recipe entropy (lower is better): {best_recipe_entropy}")
    print(f"Recipe enjoyment: {best_recipe_enjoyment:.2f}")
    print()


if __name__ == "__main__":
    main()
