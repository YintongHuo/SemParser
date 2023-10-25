class Knowledge():
    def __init__(self):
        self.knowledge_pair = []
        self.knowledge_concept = []
        self.knowledge_instance = []

        self.instance2concept = {}
        self.concept2instance = {}

    def find_concept_by_instance(self, instance):
        if instance in self.instance2concept.keys():
            return self.instance2concept[instance]
        else:
            return ''

    def find_instance_by_concept(self, concept):
        if concept in self.concept2instance.keys():
            return self.concept2instance[concept]
        else:
            return ''

    def add_pair(self, concept, instance):
        if '{}$${}'.format(concept, instance) not in self.knowledge_pair:
            self.knowledge_pair.append('{}$${}'.format(concept, instance))

        if concept not in self.knowledge_concept:
            self.knowledge_concept.append(concept)

            self.concept2instance[concept] = [instance]
        else:
            if instance not in self.concept2instance[concept]:
                self.concept2instance[concept].append(instance)

        if instance not in self.knowledge_instance:
            self.knowledge_instance.append(instance)
            self.instance2concept[instance] = [concept]
        else:
            if concept not in self.instance2concept[instance]:
                self.instance2concept[instance].append(concept)