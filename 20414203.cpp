// Name: Tan Yee Yang
// StudentID: 20414203
#include <iostream>
#include <vector>
#include <cstdlib> // For atoi
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <map>

// Initialize global counter to count the occurance of different type of heursitic
int shiftCounter = 0;
int splitCounter = 0;
int exchange_largestBin_largestItemCounter = 0;
int exchange_randomBin_reshuffleCounter = 0;
int bestPackingCounter = 0;

// Define a constant seed valye for the random number generator for reproducibility
#define RANDOM_SEED 8234 // Random Seed = {6, 238, 2342, 8234, 45325}, 8234 is best seed

struct Item{
    int size;
    
    // Overload the equality operator to compare items based on their size
    bool operator==(const Item& other) const {
        return size == other.size;
    }
};

struct Bin{
    int capacity;
    std::vector<Item> items;
    int currentBinLoad = 0;

    // Constructor
    Bin(int capacity) : capacity(capacity) {}
    
    /* Helper function */
    // Function to add an item to the bin
    void addItem(const Item& item) {
        if (currentBinLoad + item.size <= capacity) {
            items.push_back(item);
            currentBinLoad += item.size;
        } else {
            // Handle the error: item is too large to fit in the bin
            std::cerr << "Error: Item size exceeds bin capacity." << std::endl;
            // Exit the program
            exit(1);
        }
    }   

    // Function to check if an item can be added without exceeding capacity
    bool canAddItem(const Item& item) const {
        return currentBinLoad + item.size <= capacity;
    }

    // Function to remove an item from the bin only once and not all the items with the same size
    void removeItem(const Item& item) {
        auto it = std::find(items.begin(), items.end(), item);
        if (it != items.end()) {
            currentBinLoad -= item.size;
            items.erase(it);
        }
    }

    // Overload the equality operator to compare bins based on their capacity, items and currentBinLoad
    bool operator==(const Bin& other) const {
        return capacity == other.capacity && items == other.items && currentBinLoad == other.currentBinLoad;
    }
};

/**
 * Struct to encode the test problem data
*/
struct TestProblem{
    std::string problemId;
    int binCapacity;
    int numItems;
    int bestKnown;
    std::vector<Item> items;
};

using Solution = std::vector<Bin>; // A solution is a vector of bins

// Utility Functions
std::vector<TestProblem> readDataFile(const std::string& dataFile){
    std::vector<TestProblem> testProblems;
    
    // Open the input file
    std::ifstream inputFile(dataFile);
    if (!inputFile.is_open()){
        // std::cerr << "Error opening input file: " << dataFile << std::endl;
        throw std::runtime_error(std::string("Error opening file: ") + dataFile);
    }

    int num_testProblems; // Number of test problems (P)
    inputFile >> num_testProblems; // >> will skip whitespaces

    testProblems.reserve(num_testProblems);

    for(int testProblem = 0; testProblem < num_testProblems; testProblem++){
        TestProblem problem;
        inputFile >> problem.problemId >> problem.binCapacity >> problem.numItems >> problem.bestKnown;

        problem.items.resize(problem.numItems); // Resize the items vector to the number of items
        for(int i = 0; i < problem.numItems; i++){
            inputFile >> problem.items[i].size; // Read the size of each item
        }
        testProblems.push_back(problem);
    }

    // Close the input file
    inputFile.close();

    return testProblems;
}

/**
 * Function to create the output file and write the total number of problems to the output file
*/
void createOutputFile(const std::string& solutionFile, const int& totalNumberOfProblems){
    std::ofstream outputFile(solutionFile, std::ios::out | std::ios::trunc);
    if (!outputFile.is_open()){
        std::cerr << "Error opening output file: " << solutionFile << std::endl;
        throw std::runtime_error(std::string("Error opening file: ") + solutionFile);
    }

    // Write the total number of problems to the output file
    outputFile << totalNumberOfProblems << std::endl;

    // Close the output file
    outputFile.close();
}

/**
 * Function to write the solution to the output file
*/
void writeSolutionToOutputFile(const std::string& solutionFile, const TestProblem& testProblem, const Solution& solution, const std::vector<int>& bestSolution_OriginalItemIndex){
    std::ofstream outputFile(solutionFile, std::ios::out | std::ios::app);
    if (!outputFile.is_open()){
        std::cerr << "Error opening output file: " << solutionFile << std::endl;
        throw std::runtime_error(std::string("Error opening file: ") + solutionFile);
    }

    outputFile << testProblem.problemId << std::endl;
    outputFile << " obj=   " << solution.size() << "   " << testProblem.bestKnown - solution.size() << std::endl; 
    int i = 0;
    for (const auto& bin : solution){
        for (const auto& item : bin.items){
            outputFile << bestSolution_OriginalItemIndex[i] << " ";
            i++;
        }
        outputFile << std::endl;
    }

    // Close the output file
    outputFile.close();
}

/**
 * Function to calculate the total size of the items
*/
int calculateItemsTotalSize(const std::vector<Item>& items){
    int totalSize = 0;
    for(const Item& item : items){
        totalSize += item.size;
    }

    return totalSize;
}

/**
 * Function to calculate the lower bound (L1) to be used for Minimum Bin Slack (MBS)
*/
int calculateLowerBound(const std::vector<Item>& items, int binCapacity){
    int totalSize = calculateItemsTotalSize(items);

    // Calculate the lower bound and round up to the nearest integer
    int lowerBound = std::ceil(static_cast<double>(totalSize) / binCapacity);
    
    return lowerBound;
}

/**
 * Function to calculate the average slack for the relaxed MBS
 * - `Slack` is the difference between the bin's capacity and the total size of the items packed in it.
*/
double calculateAverageSlack(const std::vector<Item>& items, int binCapacity){
    int totalSize = calculateItemsTotalSize(items);
    int lowerBound = calculateLowerBound(items, binCapacity);
    int totalCapacity = lowerBound * binCapacity;
    int totalSlack = totalCapacity - totalSize;

    double averageSlack = static_cast<double>(totalSlack) / lowerBound;
    
    return averageSlack;
}

/**
 * Time Bounded Relaxed MBS class is responsible for generating the initial solution
*/
class TimeBoundedRelaxedMBS{
private:
    std::vector<Item> items;
    int binCapacity;
    double averageSlack;
    double timeLimit;

public:
    TimeBoundedRelaxedMBS(const std::vector<Item>& items, int binCapacity, double averageSlack, double timeLimit)
        : items(items), binCapacity(binCapacity), averageSlack(averageSlack), timeLimit(timeLimit) {}

    Solution applyVersion1 (){
        Solution solutions;
        auto startTime = std::chrono::high_resolution_clock::now();

        /** VERSION 1 **/
        while (!items.empty()){
            auto currentTime = std::chrono::high_resolution_clock::now();
            double elapsedSeconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
            
            // Check if time limit has been reached
            if (elapsedSeconds > timeLimit) {
                break;
            }

            // Sort items in descending order of size for each iteration to always start with the largest item
            std::sort(items.begin(), items.end(), [](const Item& a, const Item& b){
                return a.size > b.size;
            });

            Bin bin{binCapacity};
            bin.addItem(items.front()); // Add the largest item to the bin
            items.erase(items.begin()); // Remove the item from the vector

            // Add items to the bin until the average slack is reached
            for(auto item = items.begin(); item != items.end(); ){
                if(bin.canAddItem(*item) && ((bin.capacity - bin.currentBinLoad - item->size) >= averageSlack)){
                    bin.addItem(*item);
                    item = items.erase(item);   // Erase the item from the vector and move to the next item
                } else {
                    ++item;
                }
            }
            solutions.push_back(bin);
        }
        return solutions;    
    }
};

/**
 * Abstract class for low level heuristics
*/
class Heuristic{
public:
    double minimumWeight; // Weight of the heuristic for short term memory implementation
    double weight; // Weight of the heuristic for short term memory implementation

    /* Counters */ 
    int acceptCounter;
    int newCounter;
    int totalCounter;

public:
    Heuristic(int numberOfHeuristics, int maxIteration)
        : acceptCounter(0), newCounter(0), totalCounter(0) {
        minimumWeight = std::min(100.0 * numberOfHeuristics / maxIteration, 0.1),
        weight = minimumWeight;
    }

public:
    virtual void applyHeuristic(Solution& solution) = 0;
    virtual std::string getHeuristicName() const = 0;
    virtual ~Heuristic() = default; // Destructor
};

/**
 * Shift Heuristic
 * - selects each item from the bin with the largest residual capacity and tries to move 
 *  the items to the rest of the bins using the best fit descent heuristic
*/
class Shift : public Heuristic{
public:
    Shift(int numberOfHeuristics, int maxIteration) 
        : Heuristic(numberOfHeuristics, maxIteration) {}

    void applyHeuristic(Solution& solution) override {
        // Find the bin with the largest residual capacity
        int largestBinIndex = 0;
        int largestResidualCapacity = 0;
        for (int i = 0; i < solution.size(); i++) {
            int residualCapacity = solution[i].capacity - solution[i].currentBinLoad;
            if (residualCapacity > largestResidualCapacity) {
                largestResidualCapacity = residualCapacity;
                largestBinIndex = i;
            }
        }

        // Try to move items from the largest bin to the rest of the bins using the First Fit Decresing (FFD) heuristic
        for (int i = 0; i < solution.size(); i++) {
            if (i != largestBinIndex) {
                Bin& sourceBin = solution[largestBinIndex];
                Bin& targetBin = solution[i];

                // Sort items in the source bin in descending order of size
                std::sort(sourceBin.items.begin(), sourceBin.items.end(), [](const Item& a, const Item& b) {
                    return a.size > b.size;
                });

                // Move items from the largest bin to the target bin using the best fit descent heuristic
                for (auto item = sourceBin.items.begin(); item != sourceBin.items.end(); ) {
                    if (targetBin.canAddItem(*item)) {
                        targetBin.addItem(*item);
                        sourceBin.removeItem(*item);

                    } else {
                        ++item;
                    }
                }
            }
        }

        // Remove the largest bin if it is empty
        if (solution[largestBinIndex].currentBinLoad == 0) {
            // Remove the empty bin from the solution
            solution.erase(solution.begin() + largestBinIndex);
        }
    }

    std::string getHeuristicName() const override {
        return "Shift";
    }
};

/**
 * Split Heuristic
 * - simply moves half the items (randomly selected from) the current bin to a new bin 
 *  if the number of items in the current bin exceeds the average item numbers per bin
*/
class Split : public Heuristic{
private:
    std::mt19937 generator;

public:
    Split(int numberOfHeuristics, int maxIteration) 
        : Heuristic(numberOfHeuristics, maxIteration), generator(RANDOM_SEED) {}

    void applyHeuristic(Solution& solution) override {
        std::uniform_int_distribution<int> distribution(0, std::numeric_limits<int>::max());

        // Check if the number of items in the current bin exceeds the average item numbers per bin
        int totalItems = 0;
        int totalBins = 0;
        for (const auto& bin : solution) {
            totalItems += bin.items.size();
            totalBins++;
        }
        int averageItemsPerBin = totalItems / totalBins;

        std::vector<Bin> newBins;
        
        // Implement the split heuristic
        for (int i = 0; i < solution.size(); i++) {
            auto& bin = solution[i];
            if (bin.items.size() > averageItemsPerBin) {
                /* Version 2 */
                // Shuffle the items
                std::shuffle(bin.items.begin(), bin.items.end(), generator);

                // Move half of the items to a new bin
                int numItemsToMove = bin.items.size() / 2;
                std::vector<Item> itemsToMove(bin.items.begin(), bin.items.begin() + numItemsToMove);

                // Remove the selected items from the current bin using removeItem
                for (const auto& item : itemsToMove) {
                    bin.removeItem(item);
                }

                // Create a new bin and move the selected items to the new bin
                Bin newBin{bin.capacity};
                for (const auto& item : itemsToMove) {
                    newBin.addItem(item);
                }

                // Add the new bin to the newBins vector
                newBins.push_back(newBin);
            }
        }

        // Append newBins to the solution
        solution.insert(solution.end(), newBins.begin(), newBins.end());

        // Remove empty bins
        for (auto it = solution.begin(); it != solution.end(); ) {
            if (it->items.empty()) {
                it = solution.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::string getHeuristicName() const override {
        return "Split";
    }
};

/**
 * Exchange largest bin largest item Heuristic
 * - Selects the largest item from the bin with the largest residual capacity and exchanges 
 *  this item with another smaller item (or several items whose capacity sum is smaller) from 
 *  another randomly selected non-fully-filled bin
 * - The idea behind this heuristic is to transfer smaller residual capacity from a random bin to 
 *  a bin with the largest residual capacity so that this bin can be emptied by other heuristic(s).
*/
class Exchange_largestBin_largestItem : public Heuristic{
private:
    std::mt19937 generator;

public:
    Exchange_largestBin_largestItem(int numberOfHeuristics, int maxIteration) 
        : Heuristic(numberOfHeuristics, maxIteration), generator(RANDOM_SEED) {}

    void applyHeuristic(Solution& solution) override {
        // Find the bin with the largest residual capacity
        int largestResidualCapacityBinIndex = 0;
        int largestResidualCapacity = 0;
        for (int i = 0; i < solution.size(); i++) {
            int residualCapacity = solution[i].capacity - solution[i].currentBinLoad;
            if (residualCapacity > largestResidualCapacity) {
                largestResidualCapacity = residualCapacity;
                largestResidualCapacityBinIndex = i;
            }
        }

        // Select the largest item from the bin with the largest residual capacity
        Bin& largestBin = solution[largestResidualCapacityBinIndex];
        if (largestBin.items.empty()) {
            return; // Skip empty bins
        }

        Item largestItem = largestBin.items.front();
        for (int i = 1; i < largestBin.items.size(); i++) {
            if (largestBin.items[i].size > largestItem.size) {
                largestItem = largestBin.items[i];
            }
        }

        // Find all bin that is not fully filled
        std::vector<int> nonFullyFilledBins;
        for (int i = 0; i < solution.size(); i++) {
            if (solution[i].currentBinLoad < solution[i].capacity) {
                nonFullyFilledBins.push_back(i);
            }
        }

        // If there are no non-fully-filled bins, return
        if (nonFullyFilledBins.empty()) {
            return; // No non-fully-filled bins to exchange items with
        }

        // Select a random non-fully-filled bin
        std::uniform_int_distribution<int> distribution(0, nonFullyFilledBins.size() - 1);
        int randomBinIndex = nonFullyFilledBins[distribution(generator)];

        // Find a smaller item (or several items whose capacity sum is smaller) from the random bin
        Bin& randomBin = solution[randomBinIndex];
        std::vector<Item> smallerItems;
        int smallerItemsSize = 0;
        for (int i = 0; i < randomBin.items.size(); i++) {
            if (randomBin.items[i].size < largestItem.size && randomBin.items[i].size + smallerItemsSize <= largestItem.size) {
                smallerItems.push_back(randomBin.items[i]);
                smallerItemsSize += randomBin.items[i].size;
            }
        }

        // Calculate the size available for the random bin after removing the smaller items
        int randomBinAvailableCapacity = randomBin.capacity - randomBin.currentBinLoad + smallerItemsSize;
        // If smaller items are found, exchange them with the largest item
        if (!smallerItems.empty() && randomBinAvailableCapacity >= largestItem.size) {
            // Remove largest item from the largest bin
            auto it = std::find(largestBin.items.begin(), largestBin.items.end(), largestItem);
            if (it != largestBin.items.end()) {
                largestBin.items.erase(it);
                largestBin.currentBinLoad -= largestItem.size;
            }

            for (const auto& item : smallerItems) {
                // Add the smaller item to the largest bin
                largestBin.items.push_back(item);
                largestBin.currentBinLoad += item.size;
                
                // Remove the smaller item from the random bin
                auto it2 = std::find(randomBin.items.begin(), randomBin.items.end(), item);
                if (it2 != randomBin.items.end()) {
                    randomBin.items.erase(it2);
                    randomBin.currentBinLoad -= item.size;
                }
            }

            // Add the largest item to the random bin
            randomBin.items.push_back(largestItem);
            randomBin.currentBinLoad += largestItem.size;
        }

        // Remove empty bins
        for (auto it = solution.begin(); it != solution.end(); ) {
            if (it->items.empty()) {
                it = solution.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::string getHeuristicName() const override {
        return "Exchange_largestBin_largestItem";
    }
};

/**
 * Exchange random bin reshuffle Heuristic
 * - Attempts to transfer residual capacity to the bins with larger residual capacity.
 * - The heuristic then randomly selects two non-fully-filled bins
 * - The probability of selection is proportional to the amount of their residual capacity
 * - All items from these two bins are then considered and the best items' combination is identified so that it can
 *  maximally fill one bin
 * - The remaining items are filled into other bin
*/
class Exchange_randomBin_reshuffle : public Heuristic{
private:
    std::mt19937 generator;
    
public:
    Exchange_randomBin_reshuffle(int numberOfHeuristics, int maxIteration) 
        : Heuristic(numberOfHeuristics, maxIteration), generator(RANDOM_SEED) {}

    void applyHeuristic(Solution& solution) override {
        // Find all bin that is not fully filled
        std::vector<int> nonFullyFilledBinsIndex;
        for (int i = 0; i < solution.size(); i++) {
            if (solution[i].currentBinLoad < solution[i].capacity) {
                nonFullyFilledBinsIndex.push_back(i);
            }
        }

        // If there are less than 2 non-fully-filled bins, return
        if (nonFullyFilledBinsIndex.size() < 2) {
            return; // Not enough non-fully-filled bins to reshuffle
        }

        // Initialize the probability distribution for selecting a bin
        std::vector<double> probabilities(solution.size()); // Probabilities of selecting each bin
        double totalResidualCapacity = 0.0;
    
        // Calculate the total residual capacity of all bins
        for (int i = 0; i < solution.size(); i++) {
            double residualCapacity = solution[i].capacity - solution[i].currentBinLoad;
            totalResidualCapacity += residualCapacity;
        }

        // Calculate the probability of selecting each bin
        for (int i = 0; i < solution.size(); i++) {
            int residualCapacity = solution[i].capacity - solution[i].currentBinLoad;
            double probability = residualCapacity / totalResidualCapacity;
            probabilities[i] = probability;
        }

        // Calculate sum of probabilities
        double sumProbabilities = 0.0;
        for (const auto& probability : probabilities) {
            sumProbabilities += probability;
        }

        // Select two bins based on the probability distribution
        int selectedBinIndex1 = selectRandomBin(probabilities);
        int selectedBinIndex2 = selectRandomBin(probabilities);
        while (selectedBinIndex1 == selectedBinIndex2) {
            selectedBinIndex2 = selectRandomBin(probabilities);
        }

        // Find the best items' combination of all items to maximally fill one bin
        Bin& bin1 = solution[selectedBinIndex1];
        Bin& bin2 = solution[selectedBinIndex2];
        
        // Combine all items from the two bins
        std::vector<Item> allItems;
        allItems.insert(allItems.end(), bin1.items.begin(), bin1.items.end());
        allItems.insert(allItems.end(), bin2.items.begin(), bin2.items.end());

        // Find the best items' combination to maximally fill one bin
        std::vector<Item> bestItemsCombination;
        int bestItemsCombinationSize = 0;

        // Generartea and iterate through all items to find the best items' combination
        for (size_t i = 0; i < (1 << allItems.size()); ++i){
            std::vector<Item> currentItemsCombination;
            int currentItemsCombinationSize = 0;

            for (size_t j = 0; j < allItems.size(); ++j){
                if (i & (1 << j)){
                    currentItemsCombination.push_back(allItems[j]);
                    currentItemsCombinationSize += allItems[j].size;
                }
            }

            if (currentItemsCombinationSize <= bin1.capacity && currentItemsCombinationSize > bestItemsCombinationSize){
                bestItemsCombination.clear();
                bestItemsCombination.insert(bestItemsCombination.end(), currentItemsCombination.begin(), currentItemsCombination.end());
                bestItemsCombinationSize = currentItemsCombinationSize;
            }
        }

        // Fill the best items' combination into one of the bins
        if (!bestItemsCombination.empty()) {
            bin1.items.clear();
            bin1.currentBinLoad = 0;

            for (const auto& item : bestItemsCombination) {
                bin1.addItem(item);
                
                auto it = std::find(allItems.begin(), allItems.end(), item);
                if (it != allItems.end()) {
                    allItems.erase(it);
                }
            
            }
        }

        bin2.items.clear();
        bin2.currentBinLoad = 0;

        // Add the remaining items in allItems to bin2
        for (const auto& item : allItems) {
            bin2.addItem(item);
        }

        // Remove bin2 from solution if it's empty
        if (bin2.items.empty()) {
            solution.erase(solution.begin() + selectedBinIndex2);
        }

        // Add bin1 and bin2 back to the solution
        solution[selectedBinIndex1] = bin1;
        solution[selectedBinIndex2] = bin2;

        // Remove empty bins
        for (auto it = solution.begin(); it != solution.end(); ) {
            if (it->items.empty()) {
                it = solution.erase(it);
            } else {
                ++it;
            }
        }
    }

    /**
     * Function to select a random bin based on the probability distribution
    */
    int selectRandomBin(const std::vector<double>& probabilities) {
        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
        return distribution(generator);
    }

    std::string getHeuristicName() const override {
        return "Exchange_randomBin_reshuffle";
    }
};

/**
 * Best Packing Heuristic
 * - Firstly selects the biggest item from a probalilistically selected bin
 * - The time bounded relaxed MBS heuristic is then used to search for a good packing that contains
 *  this item and considers all the other items (the sequence of these items is sorted by the residual capacity
 *  of the corresponding bins with ties broken arbitrarily)
 * - All the items that appeared in the packing found by the time bounded relaxed MBS heuristic are then transferred
 *  into a new bin
 * - Time limit is set to 0.02 seconds for quick implementation of the heuristic
 * - The probability of selecting a bin is calculated by:
 *  prob(i) = residualCapacity(i) / totalResidualCapacity
 * - The selection is in favour of bins with larger residual capacities
*/
class BestPacking : public Heuristic{
private:
    std::mt19937 generator;
    const double TIME_LIMIT = 0.02; // Time limit for the heuristic
    
public:
    BestPacking(int numberOfHeuristics, int maxIteration) 
        : Heuristic(numberOfHeuristics, maxIteration), generator(RANDOM_SEED) {}

    void applyHeuristic(Solution& solution) override {
        // Initialize the probability distribution for selecting a bin
        std::vector<double> probabilities(solution.size()); // Probabilities of selecting each bin
        double totalResidualCapacity = 0.0;
    
        // Calculate the total residual capacity of all bins
        for (int i = 0; i < solution.size(); i++) {
            double residualCapacity = solution[i].capacity - solution[i].currentBinLoad;
            totalResidualCapacity += residualCapacity;
        }

        // Calculate the probability of selecting each bin
        for (int i = 0; i < solution.size(); i++) {
            int residualCapacity = solution[i].capacity - solution[i].currentBinLoad;
            double probability = residualCapacity / totalResidualCapacity;
            probabilities[i] = probability;
        }

        // Select a bin based on the probability distribution
        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
        int selectedBinIndex = distribution(generator);

        // Select the largest item from the selected bin
        Bin& selectedBin = solution[selectedBinIndex];
        Item largestItem = selectedBin.items.front();
        for (int i = 1; i < selectedBin.items.size(); i++) {
            if (selectedBin.items[i].size > largestItem.size) {
                largestItem = selectedBin.items[i];
            }
        }

        Bin newBin(selectedBin.capacity);

        // Add the largest item to the new bin
        newBin.addItem(largestItem);
        selectedBin.removeItem(largestItem);       

        // Sort the bins by their residual capacity in descending order
        std::sort(solution.begin(), solution.end(), [](const Bin& a, const Bin& b){
            return a.capacity - a.currentBinLoad > b.capacity - b.currentBinLoad;
        });

        // Apply the time bounded relaxed MBS heuristic
        modifiedFirstFitDecreasing(solution, selectedBin, largestItem, newBin);

        // Add the new bin to the solution
        solution.push_back(newBin);

        // Remove empty bins
        for (auto it = solution.begin(); it != solution.end(); ) {
            if (it->items.empty()) {
                it = solution.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::string getHeuristicName() const override {
        return "BestPacking";
    }

    void modifiedFirstFitDecreasing(Solution& solution, Bin& selectedBin, Item& largestItem, Bin& newBin) {
        auto start = std::chrono::high_resolution_clock::now();

        // Try to add the items from the bin in solution to the new bin until the time limit is reached and skip the selectedbin
        for (int i = 0; i < solution.size(); i++) { // For each bin in the solution
            Bin& bin = solution[i];

            for (int j = 0; j < bin.items.size(); j++) {    // For each item in the bin
                const Item& item = bin.items[j];
                auto now = std::chrono::high_resolution_clock::now();
                double elapsedSeconds = std::chrono::duration<double>(now - start).count();

                if (elapsedSeconds >= TIME_LIMIT) {
                    break;
                }

                if (newBin.canAddItem(item)) {
                    newBin.addItem(item);
                    bin.removeItem(item);
                }
            }
        }
    }
};

/**
 * Simulated Annealing Hyper Heuristic
*/
class SimulatedAnnealingHyperHeuristic{
private:
    std::vector<Heuristic*> heuristics; // List of pointers to low level heuristics
    double startTemperature = 1;    // Starting temperature, ts
    double stopTemperature = 0.01;     // Stopping temperature, te
    double temperature = startTemperature; // Current temperature, t
    double improvementTemperature = startTemperature;                // Improvement temperature, timp (the temperature at which the last better solution was found)
    int maxTime = 30;   // Maximum time allowed for the hyper-heuristic to run
    int numberOfHeuristics = 5; // Number of heuristics
    
    // int maxIterations = 40000;  // Total number iteration of iteration allowed (K)
    int maxIterations = 80000;  // Total number iteration of iteration allowed (K)
    int currentIteration = 0;       // Current iteration
    int nrep = numberOfHeuristics;  // Number of iterations at each temperaturre
    double beta = (startTemperature - stopTemperature) * nrep / (maxIterations * startTemperature * stopTemperature); // Temeprature reduction rate / cooling rate
    bool fr = false; // Flag to track whether non-improving acceptance has occured and also flags the reheating phase
 
    // ! maxIteration = 40000, LP = max * 0.5, bin1 get 1, bin3 get 0.15, bin11 get 0.2

    // int learningPeriod = std::max(maxIterations / 500, numberOfHeuristics); // LP < K (Learning Period used in teacher's paper)
    int learningPeriod = std::round(maxIterations * 0.5);
    double initialNonImprovingAcceptanceRatio = 0.1;  // Initial non-improving acceptance ratio, rs
    double stoppingNonImprovingAcceptanceRatio = 0.005; // Stopping non-improving acceptance ratio, re
    int acceptedSolutionsCounter = 0; // Counter for total number of accepted heuristic calls during the current learning period, Ca
    Solution initialSolution; // Initial solution
    Solution currentSolution; // Current solution
    Solution bestSolution;    // Best solution found so far

    // Random engine and distribution for stochastic heuristics selection mechanism
    std::mt19937 generator;
    std::uniform_real_distribution<double> distribution; // Give random real numbers or continuous values from 0 to 1 with uniform probability

public:
    SimulatedAnnealingHyperHeuristic(double tempature, int maxTime, Solution solution)
    : startTemperature(temperature), initialSolution(solution), generator(RANDOM_SEED), maxTime(maxTime){
        
        distribution = std::uniform_real_distribution<double>(0.0, 1.0);

        // Instantiate the Shift heuristic and add its pointer to the list of heuristics
        Shift* shiftHeuristic = new Shift(numberOfHeuristics, maxIterations);
        heuristics.push_back(shiftHeuristic);
        Split* splitHeuristic = new Split(numberOfHeuristics, maxIterations);
        heuristics.push_back(splitHeuristic);
        Exchange_largestBin_largestItem* exchangeLargestBinLargestItemHeuristic = new Exchange_largestBin_largestItem(numberOfHeuristics, maxIterations);
        heuristics.push_back(exchangeLargestBinLargestItemHeuristic);
        Exchange_randomBin_reshuffle* exchangeRandomBinReshuffleHeuristic = new Exchange_randomBin_reshuffle(numberOfHeuristics, maxIterations);
        heuristics.push_back(exchangeRandomBinReshuffleHeuristic);
        BestPacking* bestPackingHeuristic = new BestPacking(numberOfHeuristics, maxIterations);
        heuristics.push_back(bestPackingHeuristic);
    }

    int objectiveFunction(const Solution& solution) {
        // The score is the number of used bins, which is just the size of the solution vector.
        return solution.size();
    }

    void run(){
        currentSolution = initialSolution;
        bestSolution = initialSolution;

        int currentScore = objectiveFunction(currentSolution);

        // Get the start time
        auto start = std::chrono::high_resolution_clock::now();
        while(currentIteration < maxIterations){
            // Get the current time
            auto now = std::chrono::high_resolution_clock::now();

            // Calculate the duration
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();

            // If 30 seconds have passed, break the loop
            if (duration >= maxTime) {
                break;
            }

            Heuristic* selectedHeuristic = selectHeuristic();

            Solution candidateSolution = generateCandidate(selectedHeuristic, currentSolution);

            int candidateScore = objectiveFunction(candidateSolution);
            double delta = candidateScore - currentScore;   // Difference in the evaluation function between candidate solution (s') and current solution
            currentIteration++;
            
            // ! check
            bool flagSame = false;
            if (currentSolution == candidateSolution){
                flagSame = true;
            }

            bool acceptWorseSolutionFlag = false;
            if (acceptMove(delta, temperature, candidateSolution, currentSolution, acceptWorseSolutionFlag)){
                currentSolution = candidateSolution;
                currentScore = candidateScore;

                if (currentScore < objectiveFunction(bestSolution)){
                    bestSolution = currentSolution;
                }
            }
            
            if (fr == true){
                improvementTemperature = improvementTemperature / (1 - (beta * improvementTemperature));
                temperature = improvementTemperature;
            } else if (currentIteration % nrep == 0){
                temperature = temperature / (1 + (beta * temperature));
            }

            procedureLearn(selectedHeuristic, candidateSolution, delta, flagSame, acceptWorseSolutionFlag);
        }
    }

    /**
     * Getter Method to get the best solution
    */
    Solution getBestSolution() const {
        return bestSolution;
    }

    /**
     * Getter Method to get the current solution
    */
    Solution getCurrentSolution() const {
        return currentSolution;
    }

private:
    // Select a heuristic based on the probability distribution
    Heuristic* selectHeuristic(){
        double randomValue = distribution(generator);
        double totalWeight = 0.0;

        // Calculate the total weight of all heuristics
        for (const auto& heuristic : heuristics){
            totalWeight += heuristic->weight;
        }

        double cumulativeProbability = 0.0;
        for (const auto& heuristic : heuristics){
            cumulativeProbability += heuristic->weight / totalWeight;
            if (randomValue <= cumulativeProbability){
                return heuristic; // If the random value is less than the cumulative probability, select the heuristic
            }
        }

        return nullptr;
    }

    // Generate a candidate solution, s' from the current solution, s using the selected low level heuristic
    Solution generateCandidate(Heuristic* selectedHeuristic, const Solution& currentSolution){
        Solution candidateSolution = currentSolution;
        selectedHeuristic->applyHeuristic(candidateSolution);

        // Update global counter
        if (selectedHeuristic->getHeuristicName() == "Shift"){
            shiftCounter++;
        } else if (selectedHeuristic->getHeuristicName() == "Split"){
            splitCounter++;
        } else if (selectedHeuristic->getHeuristicName() == "Exchange_largestBin_largestItem"){
            exchange_largestBin_largestItemCounter++;
        } else if (selectedHeuristic->getHeuristicName() == "Exchange_randomBin_reshuffle"){
            exchange_randomBin_reshuffleCounter++;
        } else if (selectedHeuristic->getHeuristicName() == "BestPacking"){
            bestPackingCounter++;
        }

        return candidateSolution;
    }

    // Simulated Annealing Acceptance Criterion
    bool acceptMove(double delta, double temperature, Solution candidateSolution, Solution currentSolution, bool& acceptWorseSolutionFlag){
        if (delta <= 0 && (currentSolution != candidateSolution)) return true; // Always accept if the new solution is better and current solution != candidate solution

        double probability = exp(-delta / temperature);
        if (delta > 0 && distribution(generator) < probability) {   // If the new solution is worse, accept it with a probability of exp(-delta / temperature)
            acceptWorseSolutionFlag = true;
            return true; // If random value is less than the probability, accept the move (return true)
        }

        return false; // Otherwise, reject the move (return false)
    }
    
    /**
     * Procedure Learn for Short-Term Memory Mechenism
    */
    void procedureLearn(Heuristic* selectedHeuristic, Solution candidateSolution, double delta, bool flagSame, bool acceptWorseSolutionFlag){
        selectedHeuristic->totalCounter++;

        // If the candidate solution != current solution
        if (!flagSame){
            selectedHeuristic->newCounter++;
        }

        // If the new solution is better
        if (delta < 0){     
            selectedHeuristic->acceptCounter++;
            acceptedSolutionsCounter++;
            improvementTemperature = temperature;
            fr = false;
        }

        // If the new solution is worse, accept it with a probability of exp(-delta / temperature)
        if (acceptWorseSolutionFlag){
            selectedHeuristic->acceptCounter++;
            acceptedSolutionsCounter++;
        }

        if (currentIteration % learningPeriod == 0){
            if ((acceptedSolutionsCounter / learningPeriod) < stoppingNonImprovingAcceptanceRatio){
                // Trigger the reheating strategy
                fr = true;
                improvementTemperature = improvementTemperature / (1 - beta * improvementTemperature);
                temperature = improvementTemperature;
                currentSolution = bestSolution;
            
                for (const auto& heuristic : heuristics){
                    if (heuristic->totalCounter == 0){
                        heuristic->weight = heuristic->minimumWeight;   // Set the weight to the minimum weight which is small value positive to avoid division by zero
                    } else {
                        heuristic->weight = std::max(heuristic->minimumWeight, static_cast<double>(heuristic->newCounter) / heuristic->totalCounter);
                    }

                    heuristic->acceptCounter = 0;
                    heuristic->newCounter = 0;
                    heuristic->totalCounter = 0;
                }
            } else {
                for (const auto& heuristic : heuristics){
                    if (heuristic->totalCounter == 0){
                        heuristic->weight = heuristic->minimumWeight;   // Set the weight to the minimum weight which is small value positive to avoid division by zero
                    } else {
                        heuristic->weight = std::max(heuristic->minimumWeight, static_cast<double>(heuristic->acceptCounter) / heuristic->totalCounter);
                    }
                    
                    heuristic->acceptCounter = 0;
                    heuristic->newCounter = 0;
                    heuristic->totalCounter = 0;
                }
            }
            acceptedSolutionsCounter = 0;
        }
    
    }
};


/**
 * int args: number of arguments
 * const char* argv[]: array of string representing arguments
*/
int main(int args, const char* argv[]){
    std::string dataFile, solutionFile;
    int maxTime = 0;

    /**
     * Parse command line arguments
     * -s <data_file> -o <solution_file> -t <max_time>
     * Example: ./20414203 -s data.txt -o solution.txt -t 30
     * */ 
    for(int i = 1; i < args; i++){
        std::string arg = argv[i];
        if (arg == "-s" && i + 1 < args){
            dataFile = argv[++i];
        } else if (arg == "-o" && i + 1 < args){
            solutionFile = argv[++i];
        } else if (arg == "-t" && i + 1 < args){
            maxTime = std::atoi(argv[++i]); // Convert string to integerS
        } else {
            std::cerr << "Usage: " << argv[0] << " -s <data_file> -o <solution_file> -t <max_time>" << std::endl;
            return 1;
        }
    }

    if (dataFile.empty() || solutionFile.empty() || maxTime <= 0) {
        std::cerr << "Invalid arguments." << std::endl;
        return 1;
    }

    // Call the processing function
    std::vector<TestProblem> testProblems = readDataFile(dataFile);

    // Set Timer
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    createOutputFile(solutionFile, testProblems.size());

    int totalGap = 0;
    for (int p = 0; p < testProblems.size(); p++){
        int averageSlack = calculateAverageSlack(testProblems[p].items, testProblems[p].binCapacity);
        double maxMBSTime = 0.1;

        // Apply the time bounded relaxed MBS heuristic to the current problem
        TimeBoundedRelaxedMBS timeBoundedRelaxedMBS(testProblems[p].items, testProblems[p].binCapacity, averageSlack, maxMBSTime);
        Solution initialSolution = timeBoundedRelaxedMBS.applyVersion1();

        // Instantiate the Simulated Annealing Hyper Heuristic
        SimulatedAnnealingHyperHeuristic sahh(1, maxTime, initialSolution);
        sahh.run();

        Solution bestSolution = sahh.getBestSolution(); // Get the best solution

        // Compare the score of the best solution and the best known solution
        int bestKnown = testProblems[p].bestKnown;
        int bestScore = sahh.objectiveFunction(bestSolution);
        int gap = bestScore - bestKnown;
        totalGap += gap;
        std::cout << testProblems[p].problemId << " : Best Known: " << bestKnown << ", Best Score: " << bestScore << ", Gap: " << gap << std::endl;

        // Declare a pair to store item from the final solution and its original index
        std::vector<std::pair<int, int>> bestSolution_OriginalItemIndex;    // Pair of item and its original index
        
        // Pass items from the best solution to the pair
        int j = 0;
        for (int i = 0; i < bestSolution.size(); i++) {
            for (const auto& item : bestSolution[i].items) {
                bestSolution_OriginalItemIndex.push_back(std::make_pair(item.size, j));
                j++;
            }
        }
        
        // Compare the items in the final solution with the original items vector from testproblem
        std::vector<int> originalItems;
        for (const auto& item : testProblems[p].items) {
            originalItems.push_back(item.size);
        }

        // Find the original index of the items in the final solution
        std::vector<int> finalSolution_OriginalItemIndex;
        for (int i = 0; i < bestSolution_OriginalItemIndex.size(); i++) {
            for (int j = 0; j < originalItems.size(); j++) {
                if (bestSolution_OriginalItemIndex[i].first == originalItems[j]) {
                    finalSolution_OriginalItemIndex.push_back(j);
                    originalItems[j] = -1; // Mark as used to avoid duplicates
                    break;
                }
            }
        }

        // Print final solution into the solution file
        writeSolutionToOutputFile(solutionFile, testProblems[p], bestSolution, finalSolution_OriginalItemIndex);
    }

    double averageGap = static_cast<double>(totalGap) / testProblems.size();
    // std::cout << "Average Gap: " << averageGap << std::endl;
    
    // Calculate the time taken to solve the problem
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Time taken to solve the problem: " << std::chrono::duration_cast<std::chrono::seconds>(end - begin).count() << " seconds" << std::endl;

    return 0;
}


