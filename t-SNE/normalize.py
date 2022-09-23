'''Stock prices are changing not by arbitrarty real value but by multiples of minimum 
possible price change called tick size, which is usually equals to $0.01, though it 
might be different for certain stocks and exchanges.

From the host answer we know that prices were normalized by dividing them by the 
price (WAP to be precise) at seconds_in_bucket=0 for each stock_id/time_id. So tick
 size will also be divided by this price.

Then if we find normalized tick size by taking minimum of all price changes for each 
stock_id/time_id, and assuming real tick size=$0.01, we can restore original price as:
S0 = 0.01/normalized_tick_size
'''

