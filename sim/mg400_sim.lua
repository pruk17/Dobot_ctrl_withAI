-- Offline stand-in for the Dobot MG400 Lua API.
--
-- Lets a competition program run on any plain Lua interpreter with no robot
-- attached, so the sequencing can be checked before going to the cell.
-- Time is virtual: Wait/Sleep and each move charge milliseconds to a counter
-- instead of really blocking, so a full 12-piece cycle finishes instantly.

ON  = 1
OFF = 0

local sim = {}
_G.sim = sim

sim.config = {
  start_mode  = "press",  -- "press" = momentary tap, "hold" = held down forever
  start_at    = 500,      -- ms, when the operator hits Start
  start_hold  = 300,      -- ms the button stays down in "press" mode
  reset_at    = nil,      -- ms, nil = Reset never pressed
  reset_hold  = 300,
  conveyor_ms = 800,      -- travel time pusher -> middle sensor
  move_ms     = 400,      -- charged per MovJ/MovL
  time_limit  = 300000,   -- abort after 5 virtual minutes
  spin_limit  = 20000,    -- abort after N API calls with no Wait/Sleep
  poll_ms     = 0.05,     -- time one bare API call costs, mimics a real poll
  magazine    = { "circle", "square", "circle", "square", "square", "circle",
                  "square", "circle", "circle", "square", "square", "circle" },
}

-- pin names, keep in sync with the program under test
sim.out_names = { "vacuum", "pressure", "redLED", "greenLED", "conv", "pushOb" }
sim.in_names  = { "sensorConv", "sensorMag", "start", "reset" }

local cfg = sim.config
local t, spin      = 0, 0
local outs         = {}
local mag          = {}
local belt_piece   = nil   -- part sitting on the belt, not yet at the sensor
local arrive_at    = nil   -- virtual time the part reaches the sensor
local held         = nil   -- part currently on the suction cup
local here         = nil   -- point the arm is standing on
local placements   = {}
local occupied     = {}
sim.warnings       = {}

local function log(fmt, ...)
  print(string.format("[%7.0fms] %s", t, string.format(fmt, ...)))
end

local function warn(fmt, ...)
  local msg = string.format(fmt, ...)
  sim.warnings[#sim.warnings + 1] = string.format("[%dms] %s", t, msg)
  log("!! %s", msg)
end

-- A call that only reads or writes I/O still burns a little controller time,
-- but it never yields, so it also counts towards the starvation guard.
local function touch()
  t, spin = t + cfg.poll_ms, spin + 1
  if spin > cfg.spin_limit then
    error(string.format(
      "STARVED: %d API calls without a single Wait/Sleep. On the real "..
      "controller this pegs the CPU and the Stop button stops responding.",
      cfg.spin_limit), 0)
  end
end

local function tick(ms)
  t, spin = t + ms, 0
  if t > cfg.time_limit then
    error("TIMEOUT: virtual time limit reached, the program never finished", 0)
  end
end

-- points ------------------------------------------------------------------
local function point(name)
  return { __point = true, name = name, base = name, dz = 0 }
end

for i = 1, 20 do _G["P" .. i] = point("P" .. i) end
InitialPose = point("InitialPose")

function RelPoint(p, off)
  touch()
  if type(p) ~= "table" or not p.__point then
    error("RelPoint got " .. tostring(p) .. " instead of a taught point", 0)
  end
  return { __point = true, base = p.base, dz = (p.dz or 0) + (off[3] or 0),
           name = string.format("%s%+d", p.base, (p.dz or 0) + (off[3] or 0)) }
end

-- motion ------------------------------------------------------------------
local function move(kind, p)
  if type(p) ~= "table" or not p.__point then
    error(kind .. " got " .. tostring(p) .. " instead of a taught point", 0)
  end
  log("%s  %s", kind, p.name)
  here = p
  tick(cfg.move_ms)
end

function MovJ(p) move("MovJ", p) end
function MovL(p) move("MovL", p) end
function Sync() touch() end

function Sleep(ms) tick(ms or 0) end
function Wait(ms)  tick(ms or 0) end

-- digital I/O -------------------------------------------------------------
function DO(index, status)
  touch()
  local name = sim.out_names[index] or ("DO" .. tostring(index))
  local prev = outs[index]
  outs[index] = status
  if prev == status then return end
  log("DO  %-9s %s", name, status == ON and "ON" or "OFF")

  if name == "pushOb" and status == ON then
    if #mag > 0 then
      belt_piece, arrive_at = table.remove(mag, 1), nil
      log("    -> pushed a %s onto the belt (%d left in magazine)", belt_piece, #mag)
    else
      warn("pusher fired with an empty magazine")
    end

  elseif name == "conv" and status == ON then
    if belt_piece and not arrive_at then
      arrive_at = t + cfg.conveyor_ms
    end

  elseif name == "conv" and status == OFF then
    if belt_piece and arrive_at and t < arrive_at then
      warn("conveyor stopped before the part reached the sensor")
      arrive_at = nil
    end

  elseif name == "vacuum" and status == ON then
    if held then
      warn("vacuum switched on while already carrying a %s", held)
    elseif belt_piece and arrive_at and t >= arrive_at then
      held, belt_piece, arrive_at, Object = belt_piece, nil, nil, nil
      log("    -> picked up the %s", held)
    else
      warn("vacuum switched on with nothing at the pick position")
    end

  elseif name == "pressure" and status == ON then
    if not held then
      warn("blow-off fired while not carrying anything")
    else
      local slot = here and here.base or "?"
      if occupied[slot] then
        warn("slot %s already holds a %s - parts are stacking up", slot, occupied[slot])
      end
      occupied[slot] = held
      placements[#placements + 1] = { shape = held, slot = slot, at = t }
      log("    -> released the %s into %s", held, slot)
      held = nil
    end
  end
end

function DI(index)
  touch()
  local name = sim.in_names[index] or ("DI" .. tostring(index))

  if name == "sensorMag" then
    return #mag > 0 and ON or OFF

  elseif name == "sensorConv" then
    if belt_piece and arrive_at and t >= arrive_at then
      Object = belt_piece   -- what the vision kit would have reported
      return ON
    end
    return OFF

  elseif name == "start" then
    if cfg.start_mode == "hold" then
      return t >= cfg.start_at and ON or OFF
    end
    return (t >= cfg.start_at and t < cfg.start_at + cfg.start_hold) and ON or OFF

  elseif name == "reset" then
    if not cfg.reset_at then return OFF end
    return (t >= cfg.reset_at and t < cfg.reset_at + cfg.reset_hold) and ON or OFF
  end

  return OFF
end

-- runner ------------------------------------------------------------------
function sim.run(path)
  for i, s in ipairs(cfg.magazine) do mag[i] = s end
  print(("="):rep(70))
  print("running " .. path .. "   (start_mode = " .. cfg.start_mode .. ")")
  print(("="):rep(70))

  local ok, err = pcall(dofile, path)

  print(("="):rep(70))
  if not ok then print("ABORTED: " .. tostring(err)) end
  print(string.format("virtual time   : %.1f s", t / 1000))
  print(string.format("parts placed   : %d of %d", #placements, #cfg.magazine))
  local order = {}
  for _, p in ipairs(placements) do
    order[#order + 1] = p.shape:sub(1, 2) .. "->" .. p.slot
  end
  print("placement order: " .. (#order > 0 and table.concat(order, "  ") or "(none)"))
  print(string.format("left in magazine: %d   on belt: %s   on gripper: %s",
        #mag, tostring(belt_piece), tostring(held)))
  print(string.format("warnings       : %d", #sim.warnings))
  print(("="):rep(70))
end

return sim
