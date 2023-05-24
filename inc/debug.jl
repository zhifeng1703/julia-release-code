# This is a code for debugging and displaying message. There are two macro controling the functionality: DEBUG and MSG.
# The macros are by default false. It may be set by set_DEBUG and set_MSG functions or just being set before loading
# this file. 

DEFAULT_MESSAGE_LOG_PATH = "/Log/msg.txt"
DEFAULT_DEBUGMSG_LOG_PATH = "/Log/debug_msg.txt"


if !(@isdefined DEBUG)
    DEBUG = false
end

if !(@isdefined MSG)
    MSG = false
end

function set_DEBUG(d)
    DEBUG = d
end;

function set_MSG(m)
    MSG = m
end;

function assert(boo::Function, args; debug_msg=nothing)
    # This function takes a bool function boo and terminates the execution if boo returns false.

    if !DEBUG
        return true
    else
        if !(boo(args))
            if !isnothing(debug_msg) && MSG
                println("------------------------------ERROR INFO------------------------------")
                for ii = 1:length(debug_msg)
                    display(debug_msg[ii])
                end
                println("------------------------------ERROR INFO------------------------------")
            end
            throw("Assertion!")
        end
        return true
    end
end

function assert(LHS, binary_op::Function, RHS, args_rhs=nothing; debug_msg=nothing)
    # This function terminates the executation if (LHS(kw) binary_op RHS) return false.
    # Note that LHS is by default a value, RHS can be a value or function with args listed.

    if !DEBUG
        return true
    else
        rhs = isnothing(args_rhs) ? RHS : RHS(args_rhs)

        if !(binary_op(LHS, rhs))
            if !isnothing(debug_msg) && MSG
                println("------------------------------ERROR INFO------------------------------")
                for ii = 1:length(debug_msg)
                    display(debug_msg[ii])
                end
                println("------------------------------ERROR INFO------------------------------")
            end
            println("LHS value:\t", LHS, ",\t RHS value:\t", rhs)
            throw("Assertion!")
        end
        return true
    end
end

# function msg(m, multi_msg=false)
#     if MSG
#         if multi_msg
#             for ii = 1:length(m)
#                 print(m[ii])
#             end
#         else
#             print(m)
#         end
#     end
# end

function msg(args...; header = "", delimiter = "\t", footer = "", io = stdout)
    if MSG
        print(io, header)
        for (i, arg) in enumerate(args)
            print(io, "$arg", delimiter);
        end
        print(io, footer)
    end
end

function d_msg(args...; header = "", delimiter = "\t", footer = "", io = stdout)
    if MSG && DEBUG
        print(io, header)
        for (i, arg) in enumerate(args)
            print(io, "$arg", delimiter);
        end
        print(io, footer)
    end
end

# function d_msg(m, multi_msg=false)
#     if DEBUG && MSG
#         if multi_msg
#             for ii = 1:length(m)
#                 print(m[ii])
#             end
#         else
#             print(m)
#         end
#     end
# end

# function d_display(m, multi_msg=false) where {T}
#     if DEBUG && MSG
#         if multi_msg
#             for ii = 1:length(m)
#                 display(m[ii])
#             end
#         else
#             display(m)
#         end
#     end
# end


msgln(args...;header = "", delimiter = "\t", io = stdout) = msg(args...; header = header, delimiter = delimiter, footer = "\n", io = io);
d_msgln(args...;header = "", delimiter = "\t", io = stdout) = d_msg(args...; header = header, delimiter = delimiter, footer = "\n", io = io);

function msgf(args...; header = "", delimiter = "\t", footer = "", path = DEFAULT_MESSAGE_LOG_PATH)
    ios = open(pwd() * path, "a+");
    msg(args...; header = header, delimiter = delimiter, footer = footer, io = ios);
    close(ios);
end

function d_msgf(args...; header = "", delimiter = "\t", footer = "", path = DEFAULT_DEBUGMSG_LOG_PATH) 
    ios = open(pwd() * path, "a+");
    d_msg(args...; header = header, delimiter = delimiter, footer = footer, io = ios);
    close(ios);
end

msgfln(args...; header = "", delimiter = "\t", path = DEFAULT_MESSAGE_LOG_PATH) =  msgf(args...; header = header, delimiter = delimiter, footer = "\n", path = path);
d_msgfln(args...; header = "", delimiter = "\t", path = DEFAULT_DEBUGMSG_LOG_PATH) =  d_msgf(args...; header = header, delimiter = delimiter, footer = "\n", path = path);


# msgf(args...; header = "", delimiter = "\t", footer = "", io = stdout) =  msg(args...; header = header, delimiter = delimiter, footer = footer, io = io);
# d_msgf(args...; header = "", delimiter = "\t", footer = "", io = stdout) =  d_msg(args...; header = header, delimiter = delimiter, footer = footer, io = IO);

# msgfln(args...; header = "", delimiter = "\t", io = stdout) =  msg(args...; header = header, delimiter = delimiter, footer = "\n", io = io);
# d_msgfln(args...; header = "", delimiter = "\t", io = stdout) =  d_msg(args...; header = header, delimiter = delimiter, footer = "\n", io = io);


