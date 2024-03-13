#### 1.排序函数区别 Rank() ,row_number(), dense_rank()

* Rank() 相同数据排序相同，且占位
* Row_number() 相同数据排序不重复
* dense_rank()相同数据排序相同，不占位

![img](https://pic2.zhimg.com/80/v2-9557034b251f14e3193e034193f326c5_1440w.webp)

#### 2.连续登陆/活跃统计（通用N）

* 解法一：窗口函数：
  * lead(event_date, 1, null)over()（数据上移）
  * lag() over() 数据下移
* 解法二：表自关联，关联条件做逻辑计算
* 解法三：大盘排序和分组排序做差值，根据分组聚合差值相同的条数
  * 1. 新增全量数据排序row_number() over()
  * 2. 对用户分组做排序
  * 3. 两顺序值做差值，如用户连续出现，则连续出现的几条记录的差异相同
  * 4. 对用户和差值分分组聚合group by， 筛选聚合值为目标知道的用户组

```sql
select 
	distinct num
from 
(
  select num, 
        (row_number() over(order by id) - row_number() over(partition by num order by id)) as diff
   from Logs
) t
group by 
	num,diff
having 
	count(*) >= 3
```



