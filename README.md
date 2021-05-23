<h1 align="center">Меры центральности</h1>
  
  **Мера центральности** определяет наиболее важные вершины графа. Центральность применяется для выявления наиболее влиятельных лиц в социальной сети, **ключевых узлов** инфраструктуры в интернете или городских сетей и разносчиков болезни. При анализе больших графов большое внимание уделяют различным центральностям.
  
  ### Меры центральности:

- **Degree centrality**
- **Pagerank** 
- **Eigenvector centrality** 
- **Hubs and authorities** 
 
  
  ### Degree centrality:
  
  Важность узла в графе можно анализировать разными способами. Самый простой — отсортировать участников **по количеству входящих ребер**. У кого больше — тот более важен.
  
  ### Pagerank:
  
  В поиске по интернету одним из критериев для важности страниц является PageRank.
Он вычисляется при помощи случайного блуждания по графу, где в качестве узлов — страницы, а ребро между узлами — гиперссылка с одной страницы на другую. Случайный блуждатель двигается по графу и время от времени перемещается на случайный узел и начинает блуждание заново. PageRank равен доли пребывания на каком-то узле за все время блуждания. Чем он больше, тем узел важнее.

<p align="center">
<img src="https://i.gyazo.com/f36cce4d4bba8d708155aaf9bbbba379.png">
  
  ### Eigenvector centrality
  
  В общем случае имеется много различных **собственных значений** , для которых существует ненулевой собственный вектор. Однако, из дополнительного требования, чтобы все элементы собственного вектора были неотрицательны, следует (по теореме Фробениуса — Перрона), что только **наибольшее собственное значение** приводит к желательной мере центральности
  
  <p align="center">
<img src="https://i.gyazo.com/4100c78a083418f829249f0408f83b26.png">
  
  ### Hubs and authorities
  
  **Авторитетная страница** - содержит полезную информацию.
**Хаб-страница** - это документ, содержащий ссылки на авторитетные документы.

- на хорошие авторитетные страницы переходят через хорошие хаб-страницы

![f1]

- xорошие хаб-страницы ссылаются на хорошие авторитетные страницы

![f2]

  <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/5e8aeb864d5abc574e86231979f67cb8-full.png">


  <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/320973291d2e8626d818da24b9c6519c-full.png">
  
  ### Практическая часть
  
  - Для анализа была выбрана Wikipedia на русском языке (https://ru.wikipedia.org/wiki/);
  - Был реализован краулер страниц;
  - Для каждой страницы запоминался адрес и набор ссылок на другие страницы;
  - После этого строилась разреженная квадратная матрица (**1 900 000 х 1 900 000**)

  ### Пример краулинга:
  
  'https://ru.wikipedia.org/wiki/Озон', 'https://ru.wikipedia.org/wiki/Древнегреческий_язык', 'https://ru.wikipedia.org/wiki/Аммиак', 'https://ru.wikipedia.org/wiki/Теплота_испарения', 'https://ru.wikipedia.org/wiki/SMILES', 'https://ru.wikipedia.org/wiki/Фреон', 'https://ru.wikipedia.org/wiki/Экстракорпоральная_мембранная_оксигенация', 'https://ru.wikipedia.org/wiki/ Химическая_энциклопедия', 'https://ru.wikipedia.org/wiki/Гидроксид_калия', 'https://ru.wikipedia.org/wiki/Дебай', 'https:// ru.wikipedia.org/wiki/1785_год', 'https://ru.wikipedia.org/wiki/Внутривенное_вливание', 'https://ru.wikipedia.org/wiki/ Соляная_кислота', 'https://ru.wikipedia.org/wiki/Озон', 'https://ru.wikipedia.org/wiki/Дезинфекция', 'https://ru.wikipedia.org/wiki/ Детонация', 'https://ru.wikipedia.org/wiki/Температура_кипения', 'https://ru.wikipedia.org/wiki/Фосфор', 'https://ru.wikipedia.org/ wiki/Диоксин', 'https://ru.wikipedia.org/wiki/Нитрат_аммония', 'https://ru.wikipedia.org/wiki/Молярная_теплоёмкость', 'https:// ru.wikipedia.org/wiki/Азот', 'https://ru.wikipedia.org/wiki/Хлорирование', 'https://ru.wikipedia.org/wiki/PubChem', 'https:// ru.wikipedia.org/wiki/Gemeinsame_Normdatei'

#### файл с разреженной матрицей

### Результаты вычислений

  <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/49f945b0cd27b3af9381407879393a39-full.png">
  
  <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/ff9a52dda030b6ceff30bf711d5b0707-full.png">

  <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/723a862fe01e812b1251e48038a070c1-full.png">
  
  <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/845ae5ac124a48fbb042220db08c0967-full.png">
  
   <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/7ad7736661801791f0cfe0127443c120-full.png">
  
  <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/019a2211373ad8f3c8fc87c3d96c5ebb-full.png">
  
   <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/ac0bc656edd6acd3ef1050a5ccd57121-full.png">

   <p align="center">
<img src="https://cdn1.savepice.ru/uploads/2021/3/19/4c6400f8063ae3bd5029056fc129eea1-full.png">
  
[f1]: http://chart.apis.google.com/chart?cht=tx&chl=a_i\leftarrow\sum\limits_iA_{ij}h_j
[f2]: http://chart.apis.google.com/chart?cht=tx&chl=h_i\leftarrow\sum\limits_iA_{ij}a_j


  
