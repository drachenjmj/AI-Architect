

# Optimistic locking with version number
<a name="BestPractices_OptimisticLocking"></a>

Optimistic locking is a strategy that detects conflicts at write time rather than preventing them. Each item includes a version attribute that increments with every update. When updating an item, you include a [condition expression](Expressions.ConditionExpressions.md) that checks whether the version number matches the value your application last read. If another process modified the item in the meantime, the condition fails and DynamoDB returns a `ConditionalCheckFailedException`.

## When to use optimistic locking
<a name="BestPractices_OptimisticLocking_WhenToUse"></a>

Optimistic locking is a good fit when:
+ Multiple users or processes might update the same item, but conflicts are infrequent.
+ Retrying a failed write is inexpensive for your application.
+ You want to avoid the overhead and complexity of managing distributed locks.

Common examples include e-commerce inventory updates, collaborative editing platforms, and financial transaction records.

## Tradeoffs
<a name="BestPractices_OptimisticLocking_Tradeoffs"></a>

**Retry overhead in high contention**  
In high-concurrency environments, the likelihood of conflicts increases, potentially causing higher retries and write costs.

**Implementation complexity**  
Adding version control to items and handling conditional checks adds complexity to the application logic. The AWS SDK for Java v2 Enhanced Client provides built-in support through the [`@DynamoDbVersionAttribute`](https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/ddb-en-client-extensions.html#ddb-en-client-extensions-VRE) annotation, which automatically manages version numbers for you.

## Pattern design
<a name="BestPractices_OptimisticLocking_PatternDesign"></a>

Include a version attribute in each item. Here is a simple schema design:
+ Partition key – A unique identifier for each item (for example, `ItemId`).
+ Attributes:
  + `ItemId` – The unique identifier for the item.
  + `Version` – An integer that represents the version number of the item.
  + `QuantityLeft` – The remaining inventory of the item.

When an item is first created, the `Version` attribute is set to 1. With each update, the version number increments by 1.


| ItemID (partition key) | Version | QuantityLeft | 
| --- | --- | --- | 
| Bananas | 1 | 10 | 
| Apples | 1 | 5 | 
| Oranges | 1 | 7 | 

## Implementation
<a name="BestPractices_OptimisticLocking_Implementation"></a>

To implement optimistic locking, follow these steps:

1. Read the current version of the item.

   ```
   def get_item(item_id):
       response = table.get_item(Key={'ItemID': item_id})
       return response['Item']
   
   item = get_item('Bananas')
   current_version = item['Version']
   ```

1. Update the item using a condition expression that checks the version number.

   ```
   def update_item(item_id, qty_bought, current_version):
       try:
           response = table.update_item(
               Key={'ItemID': item_id},
               UpdateExpression="SET QuantityLeft = QuantityLeft - :qty, Version = :new_v",
               ConditionExpression="Version = :expected_v",
               ExpressionAttributeValues={
                   ':qty': qty_bought,
                   ':new_v': current_version + 1,
                   ':expected_v': current_version
               },
               ReturnValues="UPDATED_NEW"
           )
           return response
       except ClientError as e:
           if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
               print("Version conflict: another process updated this item.")
           raise
   ```

1. Handle conflicts by retrying with a fresh read.

   Each retry requires an additional read, so limit the total number of retries.

   ```
   def update_with_retry(item_id, qty_bought, max_retries=3):
       for attempt in range(max_retries):
           item = get_item(item_id)
           try:
               return update_item(item_id, qty_bought, item['Version'])
           except ClientError as e:
               if e.response['Error']['Code'] != 'ConditionalCheckFailedException':
                   raise
               print(f"Retry {attempt + 1}/{max_retries}")
       raise Exception("Update failed after maximum retries.")
   ```

For Java applications, the AWS SDK for Java v2 Enhanced Client provides built-in optimistic locking support through the [`@DynamoDbVersionAttribute`](https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/ddb-en-client-extensions.html#ddb-en-client-extensions-VRE) annotation, which automatically manages version numbers for you.

For more information about condition expressions, see [DynamoDB condition expression CLI example](Expressions.ConditionExpressions.md).