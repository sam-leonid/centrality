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
<img src="https://i.gyazo.com/c70c5c84a1abe512d99947a3684b83e8.png">
  
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
<img src="https://i.gyazo.com/ef760ab72d4049e5d3745ef0f16c5f12.png">
  
  <p align="center">
<img src="https://i.gyazo.com/d415a9a4c5fe376bfba0cfb11bfd3cbe.png">

  <p align="center">
<img src="https://i.gyazo.com/8d9e792cf2c2c167347b7e3e2805200b.png">
  
  <p align="center">
<img src="https://i.gyazo.com/d699ac9eaa8f20b3d9cd467df93ce1cb.png">
  
   <p align="center">
<img src="https://i.gyazo.com/fafd4b9d75affdc840a584b9fa51847c.png">
  
  <p align="center">
<img src="https://i.gyazo.com/a1af32c4161657bdaa8b81b7e7b2b935.png">
  
   <p align="center">
<img src="https://i.gyazo.com/b892e6422478b7f133df9857afd8aeb5.png">

   <p align="center">
<img src="https://i.gyazo.com/63eebd54bcc0946b9749844d8db7e488.png">
  
[f1]: http://chart.apis.google.com/chart?cht=tx&chl=a_i\leftarrow\sum\limits_iA_{ij}h_j
[f2]: http://chart.apis.google.com/chart?cht=tx&chl=h_i\leftarrow\sum\limits_iA_{ij}a_j


  
