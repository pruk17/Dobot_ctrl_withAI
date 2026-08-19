-- Entry point. Run it from the repository root:
--   lua sim/run.lua                        -- default program, momentary Start
--   lua sim/run.lua sim/program_v2.lua hold
local sim = dofile("sim/mg400_sim.lua")

local program = arg and arg[1] or "sim/program_v2.lua"
if arg and arg[2] then sim.config.start_mode = arg[2] end

sim.run(program)
