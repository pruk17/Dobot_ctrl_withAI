-- convert the received codes into readable text
function to_text(received)
  local codes = received.buf
  local text = ""
  for i = 1, #codes do
    local code = codes[i]
    if code > 0 and code < 256 then
      text = text .. string.char(code)
    end
  end
  return text
end

-- open the port and wait for the AI program
local error_code, socket = TCPCreate(true, "192.168.1.6", 9000)
print("TCPCreate error_code = " .. tostring(error_code))

if error_code ~= 0 then
  print("port is busy, restart the controller")
  return
end

TCPStart(socket, 0)
print("client connected")

-- answer every message until the program is stopped
while true do
  local read_error, received = TCPRead(socket, 0)

  if read_error == 0 and received ~= nil then
    local message = to_text(received)
    print("received [" .. message .. "]")

    if string.find(message, "Ready") then
      TCPWrite(socket, "ACK", 0)
    else
      -- pick and place logic goes here
      TCPWrite(socket, "Done", 0)
    end
  end

  -- pick up the AI program again if it disconnected and came back
  TCPStart(socket, 20)
  Sleep(20)
end
