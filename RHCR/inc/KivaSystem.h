#pragma once
#include "BasicSystem.h"
#include "KivaGraph.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class KivaSystem :
	public BasicSystem
{
public:
	KivaSystem(const KivaGrid& G, MAPFSolver& solver);
	~KivaSystem();

	json simulate(int simulation_time);


private:
	const KivaGrid& G;
	unordered_set<int> held_endpoints;
	std::vector<string> next_goal_type;

	void initialize();
	void initialize_start_locations();
	void initialize_goal_locations();
	void update_goal_locations();
	int gen_next_goal(int agent_id, bool repeat_last_goal=false);
};

