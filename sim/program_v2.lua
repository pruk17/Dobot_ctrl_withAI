-- Version: Lua 5.3.5

-- Output
local vacuum   = 1
local pressure = 2
local redLED   = 3
local greenLED = 4
local conv     = 5
local pushOb   = 6

-- Input
local sensorConv = 1
local sensorMag  = 2
local start      = 3
local reset      = 4

-- Setup
DO(vacuum, OFF)
DO(pressure, OFF)
DO(redLED, ON)
DO(greenLED, OFF)
DO(conv, OFF)
DO(pushOb, OFF)

local function pull()
    DO(vacuum, ON)
    DO(pressure, OFF)
    Wait(200)
end

local function push()
    DO(vacuum, OFF)
    DO(pressure, ON)
    Wait(200)
    DO(pressure, OFF)
end

local function vision()
    -- Dobot Vision Kit return "circle" or "square"
    return Object
end

local offset_up = {0, 0, 50, 0}

local function pickAndPlace(pickPos, dropPos)
    local upperPick = RelPoint(pickPos, offset_up)
    local upperDrop = RelPoint(dropPos, offset_up)

    MovJ(InitialPose)
    MovJ(upperPick)
    MovL(pickPos)
    pull()
    MovL(upperPick)
    MovJ(InitialPose)
    MovJ(upperDrop)
    MovL(dropPos)
    push()
    MovL(upperDrop)
    MovJ(InitialPose)
    Wait(200)
end

local dropPoints = {
    circle = {P1, P2, P3, P4, P5, P6},
    square = {P7, P8, P9, P10, P11, P12},
}

local pickPos = P13

-- Main Loop
local running = false
local count = 0
local n = 1
local m = 1

while true do
    if DI(start) == ON and count < 12 then
        DO(redLED, OFF)
        DO(greenLED, ON)
        running = true

        if DI(sensorMag) == ON then
            DO(pushOb, ON)
            Wait(200)
            DO(pushOb, OFF)

            DO(conv, ON)
            print("Conveyor ON", count + 1)

            while DI(sensorConv) == OFF do
                Wait(50)
            end

            DO(conv, OFF)
            print("Conveyor OFF")

            local object = vision()

            if object == "circle" then
                local dropPos = dropPoints.circle[n]
                pickAndPlace(pickPos, dropPos)
                n = n + 1
            elseif object == "square" then
                local dropPos = dropPoints.square[m]
                pickAndPlace(pickPos, dropPos)
                m = m + 1
            else
                print("unknown object")
            end

            count = count + 1
        end

        if count >= 12 then
            DO(greenLED, OFF)
            DO(redLED, ON)
            running = false
            print("complete: 12/12")
        end

    elseif DI(reset) == ON then
        DO(redLED, OFF)
        DO(greenLED, OFF)
        DO(conv, OFF)
        DO(pushOb, OFF)
        count = 0
        n = 1
        m = 1
        running = false
        print("System reset")

    elseif not running then
        DO(redLED, ON)
        DO(greenLED, OFF)
        -- print("Wait press Start Button")
    end
end
